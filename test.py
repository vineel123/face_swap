class a:
	def __init__(self):
		self.print_mes()
	def print_mes(self):
		print("a")

class b(a):
	def __init__(self):
		super().__init__()
	def print_mes(self):
		print("b")


a = a()
b = b()
