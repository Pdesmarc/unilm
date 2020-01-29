import wget
import os
import urllib
url="https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py"
wd = os.getcwd()

#filename = wget.download(url, out=wd)


filename = urllib.request.urlretrieve(url,"evaluate-v1.1.py")

print(filename)
