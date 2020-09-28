import omero
from omero.gateway import BlitzGateway

def refresh_omero_session(conn,user,pw):
    if conn==None:
        USERNAME = user
        PASSWORD = pw
        HOST = "134.76.18.202"
        PORT=   4064

        print("Connected.")
        conn = BlitzGateway(USERNAME, PASSWORD,host=HOST, port=PORT)
    else:
        
        USERNAME = user
        PASSWORD = pw
        HOST = "134.76.18.202"
        PORT=   4064


        print("Connected.")
        conn = BlitzGateway(USERNAME, PASSWORD,host=HOST, port=PORT)
    conn.connect()
    print(conn.isConnected())
    return conn

