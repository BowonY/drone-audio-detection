import pymongo
#connection = pymongo.MongoClient("52.176.51.56", 27017)

def db_connection(ip, port):

	connection = pymongo.MongoClient(ip, port)
	db = connection.testDB
	collection = db.student

def db_select():

	print(collection)
	print( collection.find().sort({_id:1}))

def test(ip, port):

	connection = pymongo.MongoClient(ip, port)
	db = connection.testDB
	collection = db.student
	
	return collection.find_one()

element = test("localhost", 27017)
print( element["student_id"])


