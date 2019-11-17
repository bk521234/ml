import scrapy
import shutil
import requests

import fr

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://finance.yahoo.com/',
            'https://finance.yahoo.com/m/b9b3ec85-93a1-31a5-92ac-f463bac5076b/roth-ira-contributions-with.html',
            "https://finance.yahoo.com/quote/BRK-B?p=BRK-B&.tsrc=fin-srch",
            "https://finance.yahoo.com/quote/MSFT?p=MSFT&.tsrc=fin-srch",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        #page = response.url.split("/")[-2]
        #filename = 'quotes-%s.html' % page
        #with open(filename, 'wb') as f:
        #    f.write(response.body)
        #self.log('Saved file %s' % filename)
        images_list = response.css('img')
        for img in images_list:
            src_link = img.css('img::attr(src)').get()
            self.log('Source link: {}'.format(src_link))
            response = requests.get(src_link, stream=True)
            with open('img.png', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
            fr.my_facial_recognition_matcher('img.png')

