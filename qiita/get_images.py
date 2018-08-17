# -*- coding: utf8 -*-

import requests
import re
import sys
from bs4 import BeautifulSoup
from PIL import Image
from StringIO import StringIO
import random
import os


def get_images_sync(q):
    url = 'http://www.bing.com/images/search?q='+q
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img", src=re.compile(r'http://tse\d.mm.bing.net*'))
    for i, t in enumerate(img_tags):
        src_html = t.attrs["src"]
        img = requests.get(src_html)
        j = Image.open(StringIO(img.content)) 
        j.save("/Users/dir/Workspace/nlp/img"+str(i)+".jpeg")
    
    return len(img_tags)

def get_image_async(q, first, folder):
    url = 'http://www.bing.com/images/async?q='+q+'&async=content&first='+str(first)+'1&IID=images.1'
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("a")
    for i, a in enumerate(img_tags):
        id = a.get("ihk")
        if id:
            host_num = random.randint(1,4)
            image_url = "http://tse"+str(host_num)+".mm.bing.net/th?id="+id
            img = requests.get(image_url)
            if img.content:
                j = Image.open(StringIO(img.content)) 
                j.save(folder + "/img"+str(i+first)+".jpeg")
    return len(img_tags)



if __name__ == '__main__':
    #total = get_images_sync()
    #print total

    params = sys.argv
    if len(params) == 1: 
        print "require query word and total."
    elif len(params) == 2:
        print "require total."
    elif len(params) == 3:
        q = params[1]
        req_total = int(params[2])
        first = 0
        folder = os.getcwd() + "/image"

        total = first
        while total < req_total:
            total += get_image_async(q, total, folder)
            print total

        print "end"
    else:
        print "invalid params." 
    
    


	
