const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

// Serve static files from the project directory (where `hello.html` and `index.html` are located)
app.use(express.static(__dirname));

// Database connection
const db = mysql.createConnection({
  host: 'database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com',
  user: 'admin',
  password: 'Nbmqyq17!',
  database: 'mysqlTutorial'
});

db.connect((err) => {
  if (err) throw err;
  console.log('Connected to database');
});

// Endpoint to save user data
app.post('/saveUser', (req, res) => {
  const { name, email, sub } = req.body;
  const sql = 'INSERT INTO users_test (name, email, auth0_id) VALUES (?, ?, ?)';
  db.query(sql, [name, email, sub], (err, result) => {
    if (err) {
      console.error(err);
      return res.status(500).send('Error saving user to database');
    }
    res.send('User saved');
  });
});

app.listen(5500, () => {
  console.log('Server running on http://localhost:5500');
});
