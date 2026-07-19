# MlPracticeProjects

Things that I've learned from each projects are linked and written in Farsi.

## Classification Projects

- [Sonar Classification](https://pouniq.github.io/MlPracticeProjects_reports/SC_index.html)
- [breast cancer classification](https://pouniq.github.io/MlPracticeProjects_reports/)
- [Fraud Detection classification](https://pouniq.github.io/MlPracticeProjects_reports/CCR_index.html)


### خلاصه نکات:
- کلاس های imbalanced خیلی تاثیرگذار هستند و باید بتونی آنها را کنترل کنی.
- استفاده از Pipeline ها بسیار کمک کننده و الزامی هست در پروژه ها.
- سیستم ها و افزایش قیمت GPU ها رو الان بسیار خوب درک می کنم.
- به فکر این هستم که کدها رو توی google colab بزنم به خاطر وجود GPU های قوی تری که وجود داره.



## Regression Projects



- [Regression House Price](https://pouniq.github.io/MlPracticeProjects_reports/HPR_index.html)
- [Insurance Charges regression](https://pouniq.github.io/MlPracticeProjects_reports/MI_index.html)


### خلاصه نکات:
- رگرسیون به صورت خیلی خوبی بررسی شد.
- مدل HistBoostingGradientRegressor بهترین مدل بین بقیه مدل ها انتخاب شد.
- با انجام HyperParameter Tuning تونستم بهترین ترکیب پارامتر ها رو برای مدل خودم بین بقیه ترکیب پارامترهای **موجود** پیدا کنم.
- استفاده از transformedtargetcolumn می توانیم مشکل log رو حل کنیم و با استفاده از همین تابع مقادیر پیشبینی شده را به حالت عادی برگردانیم دیگر کارها نیاز نیست جداگانه انجام شود.
- اول از همه انتخاب مدل انجام می شود، سپس بهترین مدل انتخاب شد، پارامترهای آن مدل با استفاده از gridsearch یا randomizedsearch بهترین پارامترهای ممکن را برای بهترین مدل پیدا کنیم.
- می توانیم یک تابعی بسازیم که با دادن اطلاعات پیشبینی کند.




## Clustring Projects

- [Customer Segmentation](https://pouniq.github.io/MlPracticeProjects_reports/CSC_index.html)



### خلاصه نکات:
- یکی از روش های قدرتمند یادگیری ماشین، بحث unsupervised learning هست که یکی از تسک های پرکاربرد این روش clustring هست.
- تعیین و ارزیابی خوب بودن مدل clustring توسط معیارهای متفاوتی تعیین می شود یکی از آنها silhouette score است.
  - این روش ارزیابی مدل های clustring مانند kmeans کاری که انجام می دهد به این شیوه است:
  - اول از همه فاصله بین خود هر cluster را اندازه گیری می کند و بعد از فاصله هر کلاستر را نسبت به کلاسترهای دیگر اندازه گیری می کند و یک عددی بین ۱- تا +۱ به ما می دهد.





## NLP projects :




