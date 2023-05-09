#[cfg(test)]
mod tests {
    use rusqlite::Connection;

    #[test]
    fn test_mvcc() {
        let conn = Connection::open_in_memory().unwrap();

        conn.execute("CREATE TABLE t(id) WITH mvcc", ()).unwrap();
        for _ in 1..=1024 {
            conn.execute("INSERT INTO t(id) VALUES (42)", ()).unwrap();
        }

    }
}
