# How to use:

This project is meant to be viewed in its jupyter notebook, as it is a visual project. If you wish to run the the project on your computer, first clone the directory to your device. You will need to specify the user, password, host, port, and database in the function call in `bitcoin-proj.py`: 

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
