# load library  

from bs4 import BeautifulSoup

# create some HTML code

html = "<div clas='full_name'><span style = 'font-weight:bold'>Masego</span> Azra</div>"

# parse html
soup = BeautifulSoup(html, "lxml")

# Find the div with the class "full_name"m show text
soup.find("div", {"class" : "full_name"}).text