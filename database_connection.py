import sqlite3
db = sqlite3.connect("db",check_same_thread = False)

connection = None
try:
    connection = db.cursor()
    print('Connection established')
except Exception as e:
    print('Connection Failed')


def create_user():
    connection.execute('''CREATE TABLE IF NOT EXISTS Data(
        FirstName text NOT NULL,
        LastName text,
        emailaddress varchar NOT NULL UNIQUE,
        Username varchar NOT NULL UNIQUE,
        password varchar NOT NULL
    )''')
    db.commit()


def add_user_data(first_name, last_name, emailaddress, user, password):
    connection.execute(
        '''INSERT INTO Data(FirstName,LastName,emailaddress,Username,password) values (?,?,?,?,?);''',
        (first_name, last_name, emailaddress, user, password))
    db.commit()


def display_data():
    connection.execute('''SELECT * FROM Data''')
    rows = connection.fetchall()

    return rows


def delete_data():
    connection.execute('''delete from Data''')
    print('Deleted Successfully')
    db.commit()


def column_data(column_name=None):
    if column_name == 'emailaddress':
        connection.execute('''select emailaddress from Data''')
        col = connection.fetchall()
        return col
    elif column_name == 'username':
        connection.execute('''select Username from Data''')
        col = connection.fetchall()

        return col
    else:
        connection.execute('''select * from Data''')
        rows = connection.fetchall()

        return rows


def two_columns_retrieval(user, passe):
    connection.execute('''select * from Data where Username=%s and password=%s''', (user, passe))
    rows = connection.fetchall()

    return rows

