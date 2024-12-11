query_engine = index.as_query_engine()
res = query_engine.query("In what year was the college of engineering established at the University of Notre Dame?")
print(res)