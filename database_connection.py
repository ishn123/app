import psycopg2 as pg


db = pg.connect(host='ec2-34-254-69-72.eu-west-1.compute.amazonaws.com',
                database='de99tfrt5clchm',
                user='hpunmriljsjtiw',
                password='2f4f05495984ba1eb8c18bf65510805785c22fe88a5daa0f71eb90b915e5e668')

connection = db.cursor()
if connection:
    print('Connection established')
else:
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
def add_user_data(first_name,last_name,emailaddress,user,password):
    connection.execute('''INSERT INTO Data(FirstName,LastName,emailaddress,Username,password) values (%s,%s,%s,%s,%s)''',(first_name,last_name,emailaddress,user,password))
    db.commit()
def display_data():
    connection.execute('''SELECT * FROM Data''')
    rows = connection.fetchall()
    return rows
def delete_data():
    connection.execute('''delete from Data''')
    print('Deleted Successfully')
    db.commit()
def column_data(column_name = None):
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
def two_columns_retrieval(user,passe):
    connection.execute('''select * from Data where Username=%s and password=%s''',(user,passe))
    rows =  connection.fetchall()
    return rows
