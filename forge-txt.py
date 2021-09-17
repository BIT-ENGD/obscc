from bs4 import BeautifulSoup
from markdown import markdown
import re
import os

def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text




SRC="C:\\Users\\mayong\\Desktop\\md"
DST="C:\\Users\\mayong\\Desktop\\txt"


if( not os.path.exists(DST)):
    os.mkdir(DST)


files=os.listdir(SRC)

for file in files:
    fullpath=SRC+os.sep+file
    with open(fullpath,"r",encoding="utf-8") as f:
        content=f.read()
        txt=markdown_to_text(content)
        newfile=DST+os.sep+file
        newfile=newfile.replace(".md",".txt")
        with open(newfile,"w",encoding="utf-8") as f:
            f.write(txt)
