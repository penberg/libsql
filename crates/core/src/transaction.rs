use crate::{Connection, Result};
use std::ops::Deref;

pub struct Transaction {
    conn: Connection,
}

impl Transaction {
    pub fn begin(conn: Connection) -> Result<Self> {
        let _ = conn.execute("BEGIN", ())?;
        Ok(Self { conn })
    }

    pub fn commit(&self) -> Result<()> {
        let _ = self.conn.execute("COMMIT", ())?;
        Ok(())
    }

    pub fn rollback(&self) -> Result<()> {
        let _= self.conn.execute("ROLLBACK", ())?;
        Ok(())
    }
}

impl Deref for Transaction {
    type Target = Connection;

    #[inline]
    fn deref(&self) -> &Connection {
        &self.conn
    }
}