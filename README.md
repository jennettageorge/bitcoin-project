# How to use:

This project is meant to be viewed in its jupyter notebook, as it is a visual project. 


## Running the project locally: 
If you wish to run the the project on your computer, first clone the directory to your device. 

#### MacOS:

```command line
$ git clone https://github.com/jennettageorge/bitcoin-project.git
```

If you don't have postgres downloaded, you can download with HomeBrew:
* Note: you need to have postgres downloaded to import psycogp2 *


```command line
$ brew install postgresql
$ brew services start postgresql

```


To create a database and user, start up psql:

```command line
$ psql postgres
```
(might need to use `sudo -u postgres psql`).

Then inside psql, use the command:

```SQL
postgres=# CREATE DATABASE mydb;
postgres=# CREATE USER myuser WITH PASSWORD 'mypass';
postgres=# GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
```

Once you have your database, user and password set up, the program will take care of the rest of the postgres commands. You will need to specify the user, password, host, port, and database in the function call in `bitcoin-proj.py` (host and port should be correct by default if using a local instance of postgres). 

```python
connection = psycopg2.connect(user = "jenna",
                                  password = "pwd",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "coinapi")
```


Please make sure you create a user and database in which to connect to.

Then on command line,

```command line
cd bitcoin-project
python bitcoin-proj.py
