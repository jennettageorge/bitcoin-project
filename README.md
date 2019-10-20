# How to use:

This project is meant to be viewed in it's jupyter notebook as it is a visual project. If you wish to run it as a standalone python project, you will need to specify the user, password, host, port, and database in the function call

```python
connection = psycopg2.connect(user = "jenna",
                                  password = "pwd",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "coinapi")
'''

