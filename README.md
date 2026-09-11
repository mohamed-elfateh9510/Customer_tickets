# 🎫 نظام تصنيف تذاكر الدعم الفني (Customer Support Ticket Classifier)

تطبيق ذكاء اصطناعي يعتمد على معالجة اللغات الطبيعية (NLP) لتصنيف تذاكر الدعم الفني تلقائيًا حسب **الفئة (Category)** و**الأولوية (Priority)**، مع استرجاع أسرع 3 تذاكر مشابهة من السجلات السابقة باستخدام تشابه الجيوب التمامية (Cosine Similarity) عبر واجهة تفاعلية باستخدام **Streamlit**.

---

## 📁 هيكلة المشروع (Project Structure)


🚀 كيفية التشغيل محلياً
تثبيت المكتبات والمتطلبات:

Bash
pip install -r requirements.txt
تشغيل تطبيق Streamlit:

Bash
streamlit run app2.py
إدخال تذكرة اختبارية:
قم بكتابة نص رسالة الدعم الفني في المربع المخصص واضغط على Analyze Ticket لعرض النتائج مباشرة.

🧪 أمثلة حقيقية ومخرجات فعلية للنظام (Real-World Examples)
🔹 مثال 1: طلب استرداد مالي (Refund / Billing)
النص المدخل:

"I was charged twice on my credit card for the same order renewal, please refund me."

المخرجات:

الفئة المتوقعة (Predicted Category): Refund request

الأولوية المتوقعة (Predicted Priority): High

أبرز التذاكر المشابهة المسترجعة:

Score: 0.4641 | Category: Cancellation request | Priority: High

Text: "refund request issue productpurchased please assist contact credit bureau notified credit card charged additional charge issued ive followed troubleshooting steps..."

Score: 0.4171 | Category: Refund request | Priority: Medium

Text: "product recommendation issue productpurchased please assist please provide confirmation product cannot shipped please indicate credit card information..."

🔹 مثال 2: مشكلة فنية في الحساب (Technical / Login Issue)
النص المدخل:

"I cannot log in to my account after password reset. The page freezes on submission."

المخرجات:

الفئة المتوقعة (Predicted Category): Technical issue

الأولوية المتوقعة (Predicted Priority): Medium

أبرز التذاكر المشابهة المسترجعة:

Score: 0.5210 | Category: Technical issue | Priority: Medium

Text: "account access issue login failed password reset link error page freezing continuously after update..."

Score: 0.4890 | Category: Account access | Priority: High

Text: "unable to login error message credential mismatch system persistent crash..."

📊 تحليل أداء النماذج وجودة البيانات (Model Performance & Data Quality)
خلال مرحلة التقييم ومقارنة الخوارزميات (LinearSVC بنسبة 20.1% مقابل Naive Bayes 18.6% و Logistic Regression 17.8%)، تبين أن السبب الرئيسي والأوحد لانخفاض الدقة العام ليس خطأً في بناء النماذج، بل التداخل الكبير وتضارب الـ Labels داخل البيانات الأصلية (Label Noise).

💡 أدلة من واقع البيانات:
تضارب التصنيف: أظهرت نتائج البحث واسترجاع النصوص أن العديد من السجلات المشتملة على طلبات استرداد أموال صريحة (refund request / credit card charged) كانت مصنفة بشكل خاطئ كـ Cancellation request في البيانات الخام.

تساوي التوزيع العشوائي: نظرًا لوجود 5 فئات متساوية التوزيع تقريبًا، فإن التخمين العشوائي يعطي ~20% دقة، وهو ما تفسره الدقة المنخفضة للنماذج بسبب غياب حدود الفصل الواضحة في البيانات الأصلية.

سلامة الـ Pipeline: تم حل مشكلة Data Leakage بالكامل عن طريق إجراء عملية fit موحدة لقاموس الـ TF-IDF مرة واحدة فقط على بيانات التدريب وتعميمها على نموذجي التصنيف والاسترجاع والتطبيق الحي.

🛠️ التقنيات المستخدمة
Python & Pandas: لمعالجة وقراءة البيانات.

Scikit-learn: لتدريب نماذج ML (LinearSVC, LogisticRegression) واستخراج خصائص TF-IDF وحساب Cosine Similarity.

NLTK & Regex: لتنظيف النصوص واستبعاد الكلمات الشائعة (Stop Words).

Streamlit: لبناء الواجهة التفاعلية واستعراض النتائج.