# 🥗 ConsumeWise – AI-Driven Health Analysis for Packaged Products

**ConsumeWise** is an AI-powered health analysis platform that evaluates packaged food products for nutritional quality, harmful ingredients, dietary compatibility, and brand claim accuracy. Designed to promote conscious consumer choices, ConsumeWise tailors its insights to general users or personalized health profiles — enabling informed decisions in seconds.

🌐 **Live App**: [ConsumeWise on PythonAnywhere](https://consumewiseai.pythonanywhere.com)

---

## 🚀 Key Features

### 1. 🧪 Nutritional Analysis
- Detects high levels of sugar, fats, sodium, and calories
- Flags over-processed and nutrient-deficient items
- Evaluates the degree of processing and presence of artificial additives

### 2. ⚠️ Harmful Ingredient Detection
- Identifies controversial or long-term health risk components such as:
  - Trans fats, preservatives, high-fructose corn syrup, and synthetic dyes

### 3. 🥑 Diet Compatibility Check
- Checks for compatibility with common dietary plans:
  - **Keto**, **Vegan**, **Low-carb**, **Paleo**
- Highlights compliance or violation of selected diets

### 4. 🩺 Diabetes & Allergen Friendliness
- Analyzes sugar/carb levels for diabetic safety
- Detects common allergens: gluten, dairy, peanuts, soy, etc.

### 5. 🏷 Brand Claim Verification
- Cross-checks marketing claims (e.g., "low fat", "organic")
- Highlights if claims are misleading or unsupported

---

## 🧠 Technologies Used

| Layer           | Stack / Tools                                      |
|------------------|---------------------------------------------------|
| **Frontend**     | HTML, CSS, JavaScript, React.js                   |
| **Backend**      | Python, Django (web framework)                    |
| **AI/ML**        | Gemini AI model for analysis                      |
| **API**          | OpenFoodFacts API (live product data)            |
| **Deployment**   | PythonAnywhere (Django hosting)                   |

---

## 📈 Future Scope

### 🔬 Enhanced AI Accuracy & Speed
- Improve AI model to better extract and evaluate nutritional data  
- Faster processing with optimized NLP and caching

### 📸 Ingredient Extraction from Images
- Integrate OCR to scan product labels from images  
- Auto-analyze ingredients using AI post-OCR

### 🎙️ Multi-Modal Interface
- Add **voice input** and **personalized UI memory**  
- Offer chat-like health analysis and reminders

### 🧑‍⚕️ Personalized Recommendations
- Tailor insights based on user’s health profile (e.g., diabetic, vegan, allergic)  
- Recommend or flag products accordingly

### 🛒 Real-Time Market Integration
- Sync with online grocery platforms to keep product database current  
- Suggest alternatives during online shopping

---

## ✅ How to Use ConsumeWise

1. **Get Started**
   - Go to [ConsumeWise](https://consumewiseai.pythonanywhere.com) and click “Get Started”.

2. **Login or Register**
   - New user? Sign up on the registration page.
   - Returning user? Log in with your credentials.

3. **Edit Profile**
   - Add health preferences (e.g., allergies, dietary choices) in your profile settings.

4. **Analyze Product**
   - Click “Chat” in the navbar.
   - Type the product name (e.g., “Maggi noodles”) in the input field.

5. **Receive AI-Powered Health Analysis**
   - The AI model will generate a personalized report in 10–12 seconds based on your profile and the product’s ingredients.

> ℹ️ **Tip**: For best results, input only the product name. Avoid adding quantities or extra comments.

---

# ConsumeWise: AI-Driven Health Analysis

Utilizing AI to generate a comprehensive health analysis for packaged products. This analysis can be tailored to the general public or customized based on individual user needs, providing relevant nudges for healthier decisions. The core areas covered in the analysis include:

### 1. Nutritional Analysis:
- **High Presence of Unwanted Nutrients**: 
   - Evaluates levels of fats, sugar, sodium, and calories that are ideally consumed in lower quantities.
   - Flags products with excessive amounts of these nutrients as potential risks to health.

### 2. Processing & Nutrient Deficiency:
- **Degree of Processing**: 
   - Assesses how processed the product is and its impact on nutrient retention.
   - Flags nutrient-deficient products or those heavily reliant on artificial additives.
  
### 3. Harmful Ingredients:
- **Identification of Harmful Substances**: 
   - Scans for harmful or controversial ingredients such as trans fats, artificial preservatives, high levels of fructose corn syrup, etc.
   - Provides a breakdown of ingredients that could have long-term health consequences.

### 4. Diet Compliance:
- **Compatibility with Popular Diets**:
   - Cross-checks product ingredients against common dietary plans (e.g., keto, vegan, low-carb).
   - Highlights whether the product is suitable for people following specific diets.

### 5. Diabetes & Allergen Friendliness:
- **Special Dietary Considerations**:
   - Identifies if the product is safe for individuals with diabetes by analyzing sugar and carb content.
   - Flags potential allergens such as gluten, peanuts, dairy, etc., for users with food allergies.

### 6. Brand Claims Verification:
- **Validation of Product Claims**:
   - Compares the brand's health claims against factual data.
   - Highlights if any claims made on the packaging (e.g., "low fat," "organic") could be misleading based on the actual product content.

 ![WhatsApp Image 2024-10-03 at 09 05 51_0adcc15f](https://github.com/user-attachments/assets/134761b9-64d9-47aa-b6cd-2a785a70dd52)

    

You can access our website via this Link:https://consumewiseai.pythonanywhere.com

# Technologies Used in ConsumeWise

### 1. Frontend:
- **HTML, JavaScript, CSS**: 
   - For building an interactive and responsive user interface.
   - React is utilized for dynamic components and handling complex states in the frontend.

### 2. Backend:
- **Python**: 
   - Powers the backend logic, handling data processing and integration with AI models.

### 3. Web Framework:
- **Django**: 
   - A robust and scalable web framework used for backend development, routing, and managing the database.

### 4. Language:
- **Python**: 
   - The primary programming language used for both backend development and AI model integration.

### 5. AI Model:
- **Gemini**: 
   - AI model leveraged for generating health analysis of products, providing insights into nutritional content, harmful ingredients, and diet compliance.

### 6. Deployment:
- **PythonAnywhere**: 
   - The application is deployed on PythonAnywhere, the Django application ensuring scalability and accessibility.
### 6. API: 
   - Uses openFoodFacts API to fetch the data.
# Future Scope

1. **Enhanced AI Accuracy and Speed**:
   - Future iterations will focus on making the AI model more accurate in identifying nutritional content and harmful ingredients. We will also enhance the processing speed, providing users with faster insights.

2. **Photo Scanning for Ingredient Extraction**:
   - A machine learning model will be integrated to scan product photos, extract text from ingredient labels using OCR (Optical Character Recognition), and automatically send the data to the AI model for analysis. This will make the process more seamless and user-friendly.

3. **Multi-Modal Interface**:
   - Different interaction modes will be added, such as speech recognition for hands-free usage and a personalized interface that remembers previous chats. This will allow users to engage with the system through voice commands and see more health details saved in their profile.

4. **Personalized Health Recommendations**:
   - Once a user’s health profile is set up, the AI model will provide personalized product recommendations based on their specific health conditions . This tailored experience will help users make better food choices aligned with their health goals.
5.**Keeping it up to date**:
   -New products will be intoducing to the market frequently so keeping this up to date linking this to online grocery stores.


# Steps to Use ConsumeWise:

1. **Get Started**: Click on the "Get Started" button to be redirected to the login page.
2. **Login or Sign Up**:
   - If you're a registered user , use your login credentials to sign in.
   - Otherwise, you can register by creating an account on the signup page.
3. **Edit Your Profile**:
   - After logging in, you'll be redirected to the home page.
   - Navigate to your profile and click on "Edit Profile" to add your health details.
4. **Save Changes**:
   - After entering your health information, save the details.
   - Then, navigate back to the home page.
5. **Product Health Analysis**:
   - Click on "Chat" in the navbar.
   - Enter the product name in the text input field to receive a health analysis.
6. **Receive Your Personalized Analysis**:
   - The model will take around 10-12 seconds to generate a personalized health analysis based on your profile.

**Note**: For the best results, make sure to enter only the product name in the text input field. 
