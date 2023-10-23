use std::sync::Once;

use libsql_replication::replicator::Either;
#[cfg(feature = "replication")]
use libsql_replication::replicator::Replicator;
use libsql_sys::ffi;
use tokio::sync::Mutex;
use crate::replication::client::Client;
use crate::replication::local_client::LocalClient;
#[cfg(feature = "replication")]
use crate::replication::remote_client::RemoteClient;
#[cfg(feature = "replication")]
use libsql_replication::frame::FrameNo;
#[cfg(feature = "replication")]
pub use crate::replication::Frames;
use crate::{database::OpenFlags, local::connection::Connection};
use crate::{Error::ConnectionFailed, Result};

#[cfg(feature = "replication")]
pub struct ReplicationContext {
    pub replicator: Mutex<Replicator<Either<RemoteClient, LocalClient>>>,
    client: Option<Client>,
}

// A libSQL database.
pub struct Database {
    pub db_path: String,
    pub flags: OpenFlags,
    #[cfg(feature = "replication")]
    pub replication_ctx: Option<ReplicationContext>,
}

impl Database {
    /// Open a local database file.
    pub fn open<S: Into<String>>(db_path: S, flags: OpenFlags) -> Result<Database> {
        let db_path = db_path.into();

        if db_path.starts_with("libsql:")
            || db_path.starts_with("http:")
            || db_path.starts_with("https:")
        {
            Err(ConnectionFailed(format!(
                "Unable to open local database {db_path} with Database::open()"
            )))
        } else {
            Ok(Database::new(db_path, flags))
        }
    }

    #[cfg(feature = "replication")]
    pub async fn open_http_sync(
        connector: crate::util::ConnectorService,
        db_path: String,
        endpoint: String,
        auth_token: String,
    ) -> Result<Database> {
        use std::path::PathBuf;

        use crate::util::coerce_url_scheme;

        let mut db = Database::open(&db_path, OpenFlags::default())?;

        let endpoint = coerce_url_scheme(&endpoint);
        let remote = crate::replication::client::Client::new(connector, endpoint.as_str().try_into().unwrap(), auth_token).unwrap();
        let path = PathBuf::from(db_path);
        let client = RemoteClient::new(remote.clone(), &path).await.unwrap();
        let replicator = Mutex::new(Replicator::new(Either::Left(client), path, 1000).await.unwrap());

        db.replication_ctx = Some(ReplicationContext {
            replicator,
            client: Some(remote),
        });

        Ok(db)
    }

    #[cfg(feature = "replication")]
    pub fn open_local_sync(db_path: impl Into<String>, flags: OpenFlags) -> Result<Database> {
        let db_path = db_path.into();
        let mut db = Database::open(&db_path, flags)?;

        // let replicator = Replicator::new(db_path).map_err(|e| ConnectionFailed(format!("{e}")))?;
        //
        let path = PathBuf::from(db_path);
        let client = LocalClient::new(&path).await.unwrap();
        let replicator = Mutex::new(Replicator::new(Either::Right(client), path, 1000).await.unwrap());
        db.replication_ctx = Some(ReplicationContext {
            replicator,
            client: None,
        });

        Ok(db)
    }

    pub fn new(db_path: String, flags: OpenFlags) -> Database {
        static LIBSQL_INIT: Once = Once::new();

        LIBSQL_INIT.call_once(|| {
            // Ensure that we are configured with the correct threading model
            // if this config is not set correctly the entire api is unsafe.
            unsafe {
                assert_eq!(
                    dbg!(ffi::sqlite3_config(ffi::SQLITE_CONFIG_SERIALIZED)),
                    ffi::SQLITE_OK,
                    "libsql was configured with an incorrect threading configuration and
                    the api is not safe to use. Please check that no multi-thread options have
                    been set. If nothing was configured then please open an issue at:
                    https://github.com/libsql/libsql"
                );

                assert_eq!(
                    ffi::sqlite3_initialize(),
                    ffi::SQLITE_OK,
                    "libsql failed to initialize"
                );
            }
        });

        Database {
            db_path,
            flags,
            #[cfg(feature = "replication")]
            replication_ctx: None,
        }
    }

    pub fn connect(&self) -> Result<Connection> {
        Connection::connect(self)
    }

    #[cfg(feature = "replication")]
    pub fn writer(&self) -> Result<Option<crate::replication::Writer>> {
        use crate::replication::Writer;

        if let Some(ReplicationContext { client: Some(ref client), .. }) = &self.replication_ctx {
            Ok(Some(Writer { client: client.clone() }))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "replication")]
    /// Perform a sync step, returning the new replication index, or None, if the nothing was
    /// replicated yet
    pub async fn sync_oneshot(&self) -> Result<Option<FrameNo>> {
        use libsql_replication::replicator::ReplicatorClient;

        if let Some(ref ctx) = self.replication_ctx {
            let mut replicator = ctx.replicator.lock().await;
            if !matches!(replicator.client_mut(), Either::Left(_)) {
                todo!()
            }

            if let Err(_e) = replicator.replicate().await {
                todo!()
            }

            Ok(replicator.client_mut().committed_frame_no())
        } else {
            Err(crate::errors::Error::Misuse(
                "No replicator available. Use Database::with_replicator() to enable replication"
                    .to_string(),
            ))
        }
    }

    #[cfg(feature = "replication")]
    /// Sync until caught up with primary
    // FIXME: there is no guarantee this ever returns!
    pub async fn sync(&self) -> Result<Option<FrameNo>> {
        let mut previous_fno = None;
        loop {
            let new_fno = self.sync_oneshot().await?;
            tracing::trace!("New commited fno: {new_fno:?}");
            if new_fno == previous_fno {
                break;
            } else {
                previous_fno = new_fno;
            }
        }
        Ok(previous_fno)
    }

    #[cfg(feature = "replication")]
    pub async fn sync_frames(&self, frames: Frames) -> Result<Option<FrameNo>> {
        use libsql_replication::replicator::ReplicatorClient;

        if let Some(ref ctx) = self.replication_ctx {
            let mut replicator = ctx.replicator.lock().await;
            match replicator.client_mut() {
                Either::Right(c) => {
                    c.load_frames(frames);
                },
                Either::Left(_) => todo!("invalid client"),
            }
            if let Err(_e) = replicator.replicate().await {
                todo!()
            }

            Ok(replicator.client_mut().committed_frame_no())
        } else {
            Err(crate::errors::Error::Misuse(
                "No replicator available. Use Database::with_replicator() to enable replication"
                    .to_string(),
            ))
        }
    }
}
