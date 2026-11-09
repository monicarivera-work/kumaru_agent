# `ui/academic/connection.js`

The connection pill in the header and the dialog that sets the API key. The pill shows
Connected, API key needed (amber), Server unreachable or Connecting. The dialog says the
key is held in memory only and requires a key while the server is asking for one; it tests
the key against `/overview` and keeps the previous key if the new one is rejected, then
asks the entry module to bootstrap again.
