
# parser.py
import requests
from bs4 import BeautifulSoup

headers = {'Content-Type': 'text/plain; charset=UTF-8',
'Origin': 'http://search.11st.co.kr',
'Referer': 'http://search.11st.co.kr/Search.tmall?kwd=%25EC%2595%2584%25ED%2581%2590%25EC%25B2%25B5&fromACK=recent',
'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
}


urls = []

#11번가 url 28페이지
for i in range(2,3):
    urls.append('http://search.11st.co.kr/Search.tmall?kwd=%25EC%2595%2584%25ED%2581%2590%25EC%25B2%25B5&fromACK=recent#pageNum%%'+str(i)+"/")

#옥션 url 29페이지
'''
for j in range(1,30):
    urls.append('http://browse.auction.co.kr/search?keyword=%EC%95%84%ED%81%90%EC%B2%B5&itemno=&nickname=&frm=hometab&dom=auction&isSuggestion=No&retry=&Fwk=%EC%95%84%ED%81%90%EC%B2%B5&acode=SRP_SU_0100&arraycategory=&encKeyword=%EC%95%84%ED%81%90%EC%B2%B5&k=14&p='+str(j))
'''

result = []

for url in urls:
    print(url)
    res = None
    res = requests.post(url, headers=headers)
    print(res.url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    # CSS Selector를 통해 html요소들을 찾아낸다.
    if "11st" in url:
        my_titles = soup.select(
            '.benefit_tit > a'
        )
    for title in my_titles:
        # Tag안의 텍스트
        temp = title.text.replace("\n", "")
        print("temp-----", temp)
        if temp not in result:
            result.append(temp)



for dd in result:
    print(dd)