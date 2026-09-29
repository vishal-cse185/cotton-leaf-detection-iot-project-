"""
Generative AI and Agronomic Advisory Engine for Cotton Leaf Health.

Part of: IoT, DNN and Generative AI Based Cotton Leaf Disease Detection and Advisory System.

Features:
1. Google Gemini Multimodal / Text Integration (using official google-genai or REST API)
2. Comprehensive Offline Agronomic Knowledge Base fallback (zero downtime in field)
3. 5 Target Classes: Alternaria Leaf Spot, Bacterial Blight, Fusarium Wilt, Healthy Leaf, Verticillium Wilt
4. 5 Indian Agricultural Languages: English (en), Hindi (hi), Marathi (mr), Telugu (te), Gujarati (gu)
5. Structured Recommendations: Pathogen etiology, immediate actions, organic remedies, chemical dosages, cultural prevention, and farmer voice script
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdvisoryEngine")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Attempt Gemini SDK import
GEMINI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except (ImportError, Exception):
    GEMINI_AVAILABLE = False


# ==============================================================================
# OFFLINE EXPERT AGRONOMIC REPOSITORY (5 Languages x 5 Disease Conditions)
# ==============================================================================
OFFLINE_KNOWLEDGE_BASE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Alternaria_Leaf_Spot": {
        "en": {
            "disease_name": "Alternaria Leaf Spot",
            "pathogen": "Alternaria macrospora / Alternaria alternata (Fungus)",
            "severity": "Moderate to Severe",
            "symptoms": "Circular to irregular brown spots with concentric target-board rings. Spots coalesce causing extensive leaf blight, drying, and early defoliation.",
            "immediate_action": "Isolate affected zones. Prune and destroy heavily blighted lower foliage. Avoid overhead sprinkler irrigation during evening hours.",
            "organic_remedy": "Foliar spray of cold-pressed Neem Seed Kernel Extract (NSKE 5%) or 10,000 ppm Azadirachtin (2 ml/L). Apply bio-agent Pseudomonas fluorescens (2.5 kg/ha) or Trichoderma viride (10 g/L) to leaf surfaces.",
            "chemical_control": "Foliar spray of Mancozeb 75% WP @ 2.0–2.5 g/L (500–600 g/acre) OR Propiconazole 25% EC @ 1.0 ml/L OR Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L. Repeat at 12–14 day intervals.",
            "preventive_measures": "Ensure balanced Potassium (K2O) fertilization (potassium deficiency drastically increases Alternaria susceptibility). Maintain 90x60 cm plant spacing to improve canopy ventilation.",
            "voice_summary": "Alternaria Leaf Spot detected. Spray Mancozeb at 2.5 grams per liter or Neem oil at 3 ml per liter. Ensure adequate potassium fertilizer."
        },
        "hi": {
            "disease_name": "अल्टरनेरिया पत्ती धब्बा रोग (Alternaria Leaf Spot)",
            "pathogen": "अल्टरनेरिया मैक्रोस्पोरा (कवक / फफूंद)",
            "severity": "मध्यम से गंभीर",
            "symptoms": "पत्तियों पर गोलाकार या अनियमित भूरे रंग के छल्लेदार धब्बे बनते हैं। गंभीर अवस्था में पत्तियां सूखकर समय से पहले झड़ जाती हैं।",
            "immediate_action": "संक्रमित निचली पत्तियों को तोड़कर नष्ट कर दें। खेत में शाम के समय फव्वारा सिंचाई न करें।",
            "organic_remedy": "नीम का काढ़ा (NSKE 5%) या नीम का तेल (3 मिली/लीटर) छिड़कें। ट्राइकोडर्मा विरिडी या स्यूडोमोनास फ्लोरेसेंस (5-10 ग्राम/लीटर) का पर्णीय छिड़काव करें।",
            "chemical_control": "मैंकोजेब 75% WP (2.5 ग्राम/लीटर) या प्रोपिकोनाज़ोल 25% EC (1 मिली/लीटर) का छिड़काव करें। आवश्यकतानुसार 12-15 दिन बाद दोहराएं।",
            "preventive_measures": "खेत में पोटाश की संतुलित मात्रा डालें, क्योंकि पोटाश की कमी से यह रोग तेजी से फैलता है। पौधों के बीच उचित दूरी रखें।",
            "voice_summary": "कपास में अल्टरनेरिया पत्ती धब्बा रोग पाया गया है। मैंकोजेब 2.5 ग्राम प्रति लीटर या नीम तेल 3 मिली प्रति लीटर का तुरंत छिड़काव करें।"
        },
        "mr": {
            "disease_name": "अल्टरनेरिया पानांवरील ठिपके (Alternaria Leaf Spot)",
            "pathogen": "अल्टरनेरिया मॅक्रोस्पोरा (बुरशी)",
            "severity": "मध्यम ते तीव्र",
            "symptoms": "पानांवर तपकिरी रंगाचे चक्राकार ठिपके पडतात. रोग वाढल्यास पाने करपतात व गळून पडतात.",
            "immediate_action": "रोगग्रस्त पाने गोळा करून नष्ट करा. संध्याकाळी तुषार सिंचन टाळा.",
            "organic_remedy": "निमार्क (५%) किंवा निम तेल (३ मिली/लिटर) फवारा. ट्रायकोडर्मा व्हिरिडी (१० ग्रॅम/लिटर) ची फवारणी उपयुक्त ठरते.",
            "chemical_control": "मँकोझेब ७५% WP (२.५ ग्रॅम/लिटर) किंवा प्रोपिकोनाझोल २५% EC (१ मिली/लिटर) फवारावे.",
            "preventive_measures": "पोटॅश खताचा संतुलित वापर करा. झाडांमध्ये हवा खेळती राहण्यासाठी योग्य अंतर ठेवा.",
            "voice_summary": "कापूस पिकावर अल्टरनेरिया पानांवरील ठिपके रोग आढळला आहे. मँकोझेब २.५ ग्रॅम प्रति लिटर किंवा निम तेल फवारावे."
        },
        "te": {
            "disease_name": "ఆల్టర్నేరియా ఆకుమచ్చ తెగులు (Alternaria Leaf Spot)",
            "pathogen": "ఆల్టర్నేరియా శిలీంధ్రం (Fungus)",
            "severity": "మధ్యస్థం నుండి తీవ్రం",
            "symptoms": "ఆకులపై గుండ్రని ముదురు గోధుమ రంగు వలయాలతో కూడిన మచ్చలు ఏర్పడతాయి. తీవ్రత పెరిగితే ఆకులు ఎండి రాలిపోతాయి.",
            "immediate_action": "తెగులు సోకిన ఆకులను ఏరి కాల్చివేయండి. సాయంత్రం వేళల్లో తుంపర సేద్యం చేయవద్దు.",
            "organic_remedy": "వేప నూనె 3 మి.లీ/లీటర్ లేదా సుడోమోనాస్ ఫ్లోరోసెన్స్ 10 గ్రా/లీటర్ కలిపి పిచికారీ చేయండి.",
            "chemical_control": "మాంకోజెబ్ 75% WP 2.5 గ్రా/లీటర్ లేదా ప్రొపికోనజోల్ 1 మి.లీ/లీటర్ నీటిలో కలిపి పిచికారీ చేయాలి.",
            "preventive_measures": "పొటాష్ ఎరువులను సమతుల్యంగా వేయండి. గాలి వెలుతురు సరిగ్గా సోకేలా సాంద్రత పాటించండి.",
            "voice_summary": "పత్తిలో ఆల్టర్నేరియా ఆకుమచ్చ తెగులు గుర్తించబడింది. మాంకోజెబ్ 2.5 గ్రాములు లేదా వేప నూనె 3 మి.లీ పిచికారీ చేయండి."
        },
        "gu": {
            "disease_name": "અલ્ટરનેરિયા પાનનો ટપકાનો રોગ (Alternaria Leaf Spot)",
            "pathogen": "અલ્ટરનેરિયા મેક્રોસ્પોરા (ફૂગ)",
            "severity": "મધ્યમ થી ગંભીર",
            "symptoms": "પાન પર ગોળાકાર કથ્થઈ રંગના કુંડાળાવાળા ટપકા પડે છે. વધુ ઉપદ્રવમાં પાન સુકાઈને ખરી પડે છે.",
            "immediate_action": "રોગિષ્ટ પાનને તોડીને નાશ કરો. સાંજના સમયે ફુવારા પદ્ધતિથી પિયત ન આપવું.",
            "organic_remedy": "લીંબોળીનું તેલ (૩ મિલી/લિટર) અથવા ટ્રાઇકોડર્મા (૫ ગ્રામ/લિટર) નો છંટકાવ કરવો.",
            "chemical_control": "મેન્કોઝેબ ૭૫% વે.પા. (૨.૫ ગ્રામ/લિટર) અથવા પ્રોપીકોનાઝોલ (૧ મિલી/લિટર) પાણીમાં મેળવી છંટકાવ કરવો.",
            "preventive_measures": "પોટાશ ખાતર પૂરતા પ્રમાણમાં આપવું જેથી પાનની રોગપ્રતિકારક શક્તિ વધે.",
            "voice_summary": "કપાસમાં અલ્ટરનેરિયા ટપકાનો રોગ જણાયેલ છે. મેન્કોઝેબ ૨.૫ ગ્રામ પ્રતિ લિટર અથવા લીંબોળીનું તેલ છાંટો."
        }
    },
    "Bacterial_Blight": {
        "en": {
            "disease_name": "Bacterial Blight / Angular Leaf Spot",
            "pathogen": "Xanthomonas citri pv. malvacearum (Bacterium)",
            "severity": "Severe & Rapidly Spreading",
            "symptoms": "Angular, water-soaked translucent lesions delimited by leaf veins. Lesions turn reddish-brown to black. Can advance to 'black arm' on stems and water-soaked boll rot.",
            "immediate_action": "Suspend nitrogen top-dressing immediately (excess nitrogen promotes vulnerable succulent growth). Avoid field operations when foliage is wet.",
            "organic_remedy": "Spray fresh Cow Urine (5%) blended with a pinch of Asafoetida (Hing) or Bio-agent Pseudomonas fluorescens @ 5 g/L. Spray certified organic Copper Hydroxide.",
            "chemical_control": "Foliar spray of Streptocycline (Streptomycin sulphate 90% + Tetracycline hydrochloride 10%) @ 100 mg/L (1 g in 10 L water) TANK-MIXED with Copper Oxychloride 50% WP @ 2.5–3.0 g/L. Repeat in 10–12 days.",
            "preventive_measures": "Sow acid-delinted and certified disease-free seed. Treat seed with Carboxin + Thiram before planting. Burn crop residues post-harvest.",
            "voice_summary": "Bacterial Blight detected. Spray Streptocycline 1 gram in 10 liters of water mixed with Copper Oxychloride 25 grams. Stop applying excess nitrogen."
        },
        "hi": {
            "disease_name": "जीवाणु अंगमारी / कोणीय पत्ती धब्बा (Bacterial Blight)",
            "pathogen": "जैंथोमोनास सिट्री पीवी मैलवेसियरम (जीवाणु / बैक्टीरिया)",
            "severity": "गंभीर एवं तीव्र संक्रामक",
            "symptoms": "पत्तियों की नसों के बीच कोणीय, पानी जैसे भीगे हुए धब्बे बनते हैं जो बाद में गहरे भूरे या काले हो जाते हैं। तनों पर 'ब्लैक आर्म' का रूप ले सकते हैं।",
            "immediate_action": "खेत में यूरिया या नाइट्रोजन का प्रयोग तुरंत रोकें। गीली पत्तियों की स्थिति में खेत में जुताई या निराई न करें।",
            "organic_remedy": "ताजा गोमूत्र (5%) या स्यूडोमोनास फ्लोरेसेंस (5 ग्राम/लीटर) का छिड़काव करें। कॉपर हाइड्रोक्साइड का छिड़काव करें।",
            "chemical_control": "स्ट्रेप्टोसाइक्लिन (1 ग्राम प्रति 10 लीटर पानी) + कॉपर ऑक्सीक्लोराइड 50% WP (25-30 ग्राम प्रति 10 लीटर पानी) मिलाकर छिड़कें। 10-12 दिन बाद दोबारा दोहराएं।",
            "preventive_measures": "हमेशा प्रमाणित एवं तेजाब से रोआं रहित (acid-delinted) बीज ही बोएं। फसल कटाई के बाद अवशेषों को जला दें।",
            "voice_summary": "कपास में जीवाणु अंगमारी रोग पाया गया है। स्ट्रेप्टोसाइक्लिन 1 ग्राम और कॉपर ऑक्सीक्लोराइड 25 ग्राम प्रति 10 लीटर पानी में मिलाकर तुरंत छिड़कें।"
        },
        "mr": {
            "disease_name": "जिवाणूजन्य करपा / कोनीय ठिपके (Bacterial Blight)",
            "pathogen": "झँथोमोनास सिट्री (जिवाणू)",
            "severity": "अति तीव्र",
            "symptoms": "पानांच्या शिरांमुळे मर्यादित कोनीय, पाणथळ ठिपके पडतात. नंतर ते काळपट तपकिरी होतात व फांद्यांवर काळी काकडी (ब्लॅक आर्म) तयार होते.",
            "immediate_action": "युरिया (नायट्रोजन) खत देणे त्वरित थांबवा. पाने ओली असताना शेतात काम करू नका.",
            "organic_remedy": "गोमूत्र (५%) किंवा स्युडोमोनास फ्लोरेसेन्स (५ ग्रॅम/लिटर) ची फवारणी करा.",
            "chemical_control": "स्ट्रेप्टोसायक्लिन (१ ग्रॅम प्रति १० लिटर पाणी) + कॉपर ऑक्सिक्लोराईड (२५-३० ग्रॅम प्रति १० लिटर पाणी) एकत्र करून फवारणी करा.",
            "preventive_measures": "ॲसिड-डिलिंटेड बियाणे वापरावे. शेतातील पिकाचे जुने अवशेष जाळून नष्ट करावेत.",
            "voice_summary": "कापसावर जिवाणूजन्य करपा आढळला आहे. स्ट्रेप्टोसायक्लिन १ ग्रॅम आणि कॉपर ऑक्सिक्लोराईड २५ ग्रॅम १० लिटर पाण्यात मिसळून फवारा."
        },
        "te": {
            "disease_name": "బ్యాక్టీరియా ఆకుమచ్చ తెగులు (Bacterial Blight)",
            "pathogen": "జాంతోమోనాస్ బ్యాక్టీరియా (Bacterium)",
            "severity": "తీవ్రమైనది",
            "symptoms": "ఆకుల ఈనెల మధ్య కోణీయ నీటి మచ్చలు ఏర్పడి నల్లగా మారుతాయి. కొమ్మలపై నల్లటి మచ్చలు (బ్లాక్ ఆర్మ్) ఏర్పడతాయి.",
            "immediate_action": "యూరియా వేయడం వెంటనే నిలిపివేయండి. ఆకులు తడిగా ఉన్నప్పుడు చేనులో తిరగవద్దు.",
            "organic_remedy": "ఆవు మూత్రం (5%) లేదా సుడోమోనాస్ 5 గ్రా/లీటర్ పిచికారీ చేయండి.",
            "chemical_control": "స్ట్రెప్టోసైక్లిన్ 1 గ్రాము + కాపర్ ఆక్సీక్లోరైడ్ 30 గ్రాములు 10 లీటర్ల నీటిలో కలిపి పిచికారీ చేయండి.",
            "preventive_measures": "విత్తనశుద్ధి చేసిన విత్తనాలను మాత్రమే నాటండి. పంట వ్యర్థాలను కాల్చివేయండి.",
            "voice_summary": "బ్యాక్టీరియా తెగులు నివారణకు 10 లీటర్ల నీటిలో 1 గ్రాము స్ట్రెప్టోసైక్లిన్, 30 గ్రాముల కాపర్ ఆక్సీక్లోరైడ్ కలిపి పిచికారీ చేయండి."
        },
        "gu": {
            "disease_name": "જીવાણુનો ખૂણીયો ટપકાનો રોગ (Bacterial Blight)",
            "pathogen": "ઝેન્થોમોનાસ સાયટ્રી (બેક્ટેરિયા)",
            "severity": "તીવ્ર અને ઝડપી ફેલાવો",
            "symptoms": "પાનની નસો વચ્ચે ખૂણીયા પાણીપોચા ટપકાં થાય છે જે કાળા પડી જાય છે. ડાળીઓ પર કાળો કોહવારો થાય છે.",
            "immediate_action": "યુરિયા ખાતર આપવાનું તુરંત બંધ કરો. ભીના પાકમાં કામ ન કરવું.",
            "organic_remedy": "દેશી ગાયનું ગૌમૂત્ર (૫%) અથવા સ્યુડોમોનાસ ૫ ગ્રામ/લિટર છાંટવું.",
            "chemical_control": "સ્ટ્રેપ્ટોસાઇક્લિન ૧ ગ્રામ + કોપર ઓક્સીક્લોરાઇડ ૨૫ ગ્રામ પ્રતિ ૧૦ લિટર પાણીમાં મેળવી છંટકાવ કરવો.",
            "preventive_measures": "એસિડ ટ્રીટમેન્ટ કરેલ બિયારણ જ વાવવું. પાક અવશેષોનો નાશ કરવો.",
            "voice_summary": "કપાસમાં જીવાણુ ખૂણીયા ટપકા માટે ૧૦ લિટર પાણીમાં ૧ ગ્રામ સ્ટ્રેપ્ટોસાઇક્લિન અને ૨૫ ગ્રામ કોપર ઓક્સીક્લોરાઇડ છાંટો."
        }
    },
    "Fusarium_Wilt": {
        "en": {
            "disease_name": "Fusarium Wilt",
            "pathogen": "Fusarium oxysporum f. sp. vasinfectum (Soil-borne Fungus)",
            "severity": "Critical / High Mortality",
            "symptoms": "Leaves lose turgidity, turn dull yellow from bottom upwards, margins curl and dry. Vascular browning/blackening visible when stem or taproot is sliced longitudinally.",
            "immediate_action": "Immediately uproot and burn wilted plants together with their root balls. Drench soil in a 2-meter radius to halt underground fungal mycelium spread.",
            "organic_remedy": "Enrich 500 kg well-decomposed Farmyard Manure (FYM) with 5 kg Trichoderma viride or Trichoderma harzianum; broadcast at root zones. Apply Neem cake @ 250 kg/acre.",
            "chemical_control": "Soil drenching around root base with Carbendazim 50% WP @ 1.5–2.0 g/L OR Benomyl 50% WP @ 1.0 g/L (apply 200–300 ml drenching solution per plant). Spray Potassium Nitrate (13-0-45) @ 10 g/L to reduce vascular stress.",
            "preventive_measures": "Implement 3-year crop rotation with non-host cereals (maize, sorghum). Practice deep summer ploughing for solarizing resting chlamydospores. Cultivate wilt-resistant hybrids.",
            "voice_summary": "Fusarium Wilt detected. Uproot heavily wilted plants and drench surrounding soil with Carbendazim at 2 grams per liter or Trichoderma enriched manure."
        },
        "hi": {
            "disease_name": "फ्यूजेरियम उकठा / म्लानि रोग (Fusarium Wilt)",
            "pathogen": "फ्यूजेरियम ऑक्सीस्पोरम (मृदा जनित कवक)",
            "severity": "अत्यंत गंभीर (पौधा सूखने का खतरा)",
            "symptoms": "निचली पत्तियां पीली पड़कर नीचे से ऊपर की ओर सूखने लगती हैं। तने को चीर कर देखने पर अंदर भूरी या काली नसें दिखाई देती हैं।",
            "immediate_action": "मुरझाए हुए पौधों को जड़ समेत उखाड़कर तुरंत जला दें। प्रभावित स्थान के चारों ओर की मिट्टी का उपचार करें।",
            "organic_remedy": "ट्राइकोडर्मा विरिडी (5 किग्रा) को 500 किग्रा सड़ी गोबर की खाद में मिलाकर प्रति एकड़ जड़ों के पास डालें। 200 किग्रा नीम की खली डालें।",
            "chemical_control": "कार्बेंडाजिम 50% WP (2 ग्राम प्रति लीटर पानी) का घोल बनाकर पौधों की जड़ों के पास मिट्टी को तर करें (drenching)। 13-0-45 (पोटेशियम नाइट्रेट) 10 ग्राम/लीटर का छिड़काव करें।",
            "preventive_measures": "ग्रीष्मकालीन गहरी जुताई करें। फसल चक्र में मक्का या ज्वार शामिल करें। उकठा रोधी किस्में ही लगाएं।",
            "voice_summary": "कपास में उकठा रोग पाया गया है। सूखे पौधों को उखाड़कर नष्ट करें और जड़ों में कार्बेंडाजिम 2 ग्राम प्रति लीटर पानी का घोल डालें।"
        },
        "mr": {
            "disease_name": "फ्युझारियम मर रोग (Fusarium Wilt)",
            "pathogen": "फ्युझारियम ऑक्सिस्पोरम (जमीनजन्य बुरशी)",
            "severity": "अत्यंत गंभीर",
            "symptoms": "झाडांची खालची पाने पिवळी पडतात आणि झाड मलूल होऊन वाळते. खोड कापून पाहिल्यास आतील शिरा काळ्या किंवा तपकिरी दिसतात.",
            "immediate_action": "रोगट झाडे मुळासकट उपटून नष्ट करा. बाधित भागातील मातीचे निर्जंतुकीकरण करा.",
            "organic_remedy": "ट्रायकोडर्मा हरझियानम (५ किलो) चांगल्या शेणखतात मिसळून झाडांच्या बुंध्यापाशी द्या. निंबोळी पेंड वापरा.",
            "chemical_control": "कार्बेन्डाझिम ५०% WP (२ ग्रॅम/लिटर) द्रावणाने झाडाच्या मुळाशी आळवणी (Drenching) करा.",
            "preventive_measures": "उन्हाळ्यात खोल नांगरट करा. ज्वारी किंवा मक्याचे फेरपालट पीक घ्या. मर रोग प्रतिकारक वाण निवडा.",
            "voice_summary": "कापूस पिकात मर रोग आढळला आहे. बाधित झाडे उपटून टाका आणि कार्बेंडाझिम २ ग्रॅम प्रति लिटर पाणी मुळाशी टाका."
        },
        "te": {
            "disease_name": "ఫ్యుసేరియం ఎండు తెగులు (Fusarium Wilt)",
            "pathogen": "ఫ్యుసేరియం భూమిలో ఉండే శిలీంధ్రం",
            "severity": "అత్యంత ప్రమాదకరం",
            "symptoms": "క్రింది ఆకులు పసుపు రంగులోకి మారి వాడిపోతాయి. కాండం చీల్చి చూస్తే లోపలి నాళాలు గోధుమ లేదా నలుపు రంగులో కనిపిస్తాయి.",
            "immediate_action": "తెగులు సోకిన మొక్కలను వేర్లతో సహా పీకి తగలబెట్టండి. ఇతర మొక్కలకు నీరు పారకుండా చూడండి.",
            "organic_remedy": "ట్రైకోడెర్మా విరిడే 5 కేజీలను 500 కేజీల పశువుల ఎరువుతో కలిపి మొదళ్ళలో వేయండి.",
            "chemical_control": "కార్బెండజిమ్ 2 గ్రాములు ప్రతి లీటర్ నీటికి కలిపి మొక్కల మొదళ్ళలో తడపాలి (Drenching).",
            "preventive_measures": "వేసవిలో లోతు దుక్కులు దున్నండి. పంట మార్పిడి తప్పనిసరిగా పాటించండి.",
            "voice_summary": "పత్తిలో ఎండు తెగులు గుర్తించబడింది. మొక్కల మొదళ్ళలో కార్బెండజిమ్ 2 గ్రా/లీటర్ ద్రావణాన్ని పోసి తడపండి."
        },
        "gu": {
            "disease_name": "સુકારો / ફ્યુઝેરિયમ વિલ્ટ (Fusarium Wilt)",
            "pathogen": "ફ્યુઝેરિયમ ઓક્સિસ્પોરમ (જમીનજન્ય ફૂગ)",
            "severity": "અતિ ગંભીર",
            "symptoms": "નીચેના પાન પીળા પડી કરમાઈ જાય છે. થડ ચીરીને જોતાં અંદરની નસો કાળી કે કથ્થઈ દેખાય છે.",
            "immediate_action": "સુકાયેલા છોડને મૂળ સહિત ઉપાડી સળગાવી દો. મૂળ વિસ્તારમાં માવજત આપો.",
            "organic_remedy": "ટ્રાઈકોડર્મા ૫ કિલો દેશી ખાતરમાં ભેળવીને ચાસમાં આપવું. લીંબોળી ખોળનો ઉપયોગ કરવો.",
            "chemical_control": "કાર્બેન્ડાઝીમ ૫૦% વે.પા. ૨ ગ્રામ પ્રતિ લિટર પાણીમાં ઓગાળીને છોડના મૂળમાં રેડવું (ડ્રેન્ચિંગ).",
            "preventive_measures": "ઉનાળામાં ઊંડી ખેડ કરવી અને પાકની ફેરબદલી કરવી. પ્રતિકારક જાતો વાવવી.",
            "voice_summary": "કપાસમાં સુકારો રોગ જણાયો છે. સુકાયેલા છોડ ઉપાડી નાખો અને કાર્બેન્ડાઝીમ ૨ ગ્રામ પ્રતિ લિટર મૂળમાં રેડો."
        }
    },
    "Healthy": {
        "en": {
            "disease_name": "Healthy Cotton Leaf",
            "pathogen": "None (No Disease Detected)",
            "severity": "Optimal Health",
            "symptoms": "Vibrant emerald green foliage with smooth lamina, unobstructed venation, and balanced chlorophyll synthesis. No fungal lesions or bacterial blight detected.",
            "immediate_action": "Continue routine crop monitoring. Inspect leaf undersides twice weekly for early sucking pests (aphids, jassids, thrips, whiteflies).",
            "organic_remedy": "Preventative spray of 5% Neem Seed Kernel Extract (NSKE) or Panchagavya (3%) to enhance systemic plant immunity and deter vector pests.",
            "chemical_control": "No fungicide or bactericide application required. For balanced vegetative vigor, apply foliar 19:19:19 (NPK) @ 5 g/L or 1% Potassium Nitrate (13-0-45) during square formation.",
            "preventive_measures": "Maintain scheduled drip irrigation without waterlogging. Keep yellow and blue sticky traps (10 per acre) across the field border.",
            "voice_summary": "Great news! Your cotton leaf is completely healthy. Continue regular field scouting and maintain balanced nutrition."
        },
        "hi": {
            "disease_name": "स्वस्थ कपास का पत्ता (Healthy Cotton Leaf)",
            "pathogen": "कोई रोग नहीं",
            "severity": "उत्तम स्वास्थ्य",
            "symptoms": "पत्ती पूरी तरह हरी, चमकदार और रोगमुक्त है। किसी भी प्रकार के फफूंद या जीवाणु के लक्षण नहीं हैं।",
            "immediate_action": "फसल की नियमित निगरानी जारी रखें। पत्तियों के नीचे रस चूसक कीटों (माहू, हरा तेला, सफेद मक्खी) का निरीक्षण करें।",
            "organic_remedy": "रोग प्रतिरोधक क्षमता बनाए रखने के लिए 5% नीम का काढ़ा या पंचगव्य (3%) का छिड़काव करें।",
            "chemical_control": "किसी रासायनिक कवकनाशी की आवश्यकता नहीं है। पौधों की बढ़वार के लिए 19:19:19 (NPK) 5 ग्राम प्रति लीटर पानी का छिड़काव कर सकते हैं।",
            "preventive_measures": "खेत में जलभराव न होने दें। रस चूसक कीटों की निगरानी हेतु पीले चिपचिपे कार्ड (10 प्रति एकड़) लगाएं।",
            "voice_summary": "बधाई हो! आपकी कपास की पत्ती पूरी तरह स्वस्थ है। नियमित देखरेख जारी रखें और संतुलित खाद पानी दें।"
        },
        "mr": {
            "disease_name": "निरोगी कापूस पान (Healthy Cotton Leaf)",
            "pathogen": "कोणताही रोग नाही",
            "severity": "उत्तम आरोग्य",
            "symptoms": "पान पूर्णपणे हिरवेगार, तजेलदार आणि रोगमुक्त आहे. बुरशी किंवा जिवाणूचा कोणताही संसर्ग नाही.",
            "immediate_action": "नियमित पाहणी सुरू ठेवा. पानांच्या खाली रसशोषक किडींचे निरीक्षण करा.",
            "organic_remedy": "रोगप्रतिकारक शक्तीसाठी ५% निंबोळी अर्क किंवा जीवामृत फवारावे.",
            "chemical_control": "कोणत्याही औषध फवारणीची गरज नाही. चांगल्या वाढीसाठी १९:१९:१९ विद्राव्य खत ५ ग्रॅम/लिटर फवारू शकता.",
            "preventive_measures": "शेतात पाण्याचा निचरा योग्य ठेवा. पिवळे चिकट सापळे लावा.",
            "voice_summary": "छान बातमी! तुमचे कापूस पीक निरोगी आहे. नियमित खत-पाणी व्यवस्थापन सुरू ठेवा."
        },
        "te": {
            "disease_name": "ఆరోగ్యకరమైన పత్తి ఆకు (Healthy Cotton Leaf)",
            "pathogen": "ఎటువంటి తెగులు లేదు",
            "severity": "ఆరోగ్యకరం",
            "symptoms": "ఆకు పూర్తి పచ్చదనంతో ఆరోగ్యంగా ఉంది. ఎలాంటి శిలీంధ్ర లేదా బ్యాక్టీరియా మచ్చలు లేవు.",
            "immediate_action": "పైరును క్రమం తప్పకుండా గమనిస్తూ ఉండండి. రసం పీల్చే పురుగుల ఉనికిని గమనించండి.",
            "organic_remedy": "నివారణ చర్యగా 5% వేప గింజల కషాయం పిచికారీ చేయవచ్చు.",
            "chemical_control": "ఎలాంటి మందులు అవసరం లేదు. బలానికి 19:19:19 ఎరువు 5 గ్రా/లీటర్ పిచికారీ చేయవచ్చు.",
            "preventive_measures": "పొలంలో నీరు నిల్వ ఉండకుండా చూసుకోండి. పసుపు రంగు జిగురు అట్టలు అమర్చండి.",
            "voice_summary": "శుభవార్త! మీ పత్తి పైరు ఆరోగ్యంగా ఉంది. సాధారణ యాజమాన్య పద్ధతులు కొనసాగించండి."
        },
        "gu": {
            "disease_name": "તંદુરસ્ત કપાસનું પાન (Healthy Cotton Leaf)",
            "pathogen": "કોઈ રોગ નથી",
            "severity": "ઉત્તમ સ્વાસ્થ્ય",
            "symptoms": "પાન લીલુંછમ અને સંપૂર્ણ રોગમુક્ત છે. કોઈ ફૂગ કે જીવાણુની અસર નથી.",
            "immediate_action": "પાકની નિયમિત ચકાસણી ચાલુ રાખો. ચૂસિયા પ્રકારની જીવાતોનું નિરીક્ષણ કરો.",
            "organic_remedy": "રોગપ્રતિકારકતા જાળવવા ૫% લીંબોળીના અર્કનો છંટકાવ કરી શકાય.",
            "chemical_control": "કોઈ રાસાયણિક દવાની જરૂર નથી. ૧૯:૧૯:૧૯ ખાતર ૫ ગ્રામ/લિટર છાંટી શકાય.",
            "preventive_measures": "પાણીનો ભરાવો ન થવા દેવો. પીળા ચીકણા ટ્રેપ લગાવવા.",
            "voice_summary": "અભિનંદન! કપાસનું પાન એકદમ તંદુરસ્ત છે. નિયમિત માવજત ચાલુ રાખો."
        }
    },
    "Verticillium_Wilt": {
        "en": {
            "disease_name": "Verticillium Wilt",
            "pathogen": "Verticillium dahliae (Soil-borne Vascular Fungus)",
            "severity": "Moderate to High",
            "symptoms": "Classic interveinal chlorosis and necrosis creating a mottled 'tiger-stripe' pattern on leaves. Plants exhibit stunted growth, shedding of squares/bolls, and olive-brown vascular rings.",
            "immediate_action": "Rogue out early symptomatic plants. Avoid high-volume flood irrigation down rows as water streams transport infectious microsclerotia.",
            "organic_remedy": "Apply Paecilomyces lilacinus or Trichoderma viride enriched compost into the rhizosphere. Implement mustard/radish green manure bio-fumigation before sowing.",
            "chemical_control": "Soil drenching around root zone with Thiophanate Methyl 70% WP @ 1.5 g/L OR Carbendazim 12% + Mancozeb 63% WP (Saaf) @ 2.0 g/L. Foliar spray of micro-nutrients (Zinc + Boron + Magnesium) to restore vitality.",
            "preventive_measures": "Crop rotation with paddy rice (paddy submersion destroys microsclerotia) or sorghum. Avoid excessive nitrogen fertilizer; balance with potassium and phosphorus.",
            "voice_summary": "Verticillium Wilt detected with tiger-stripe leaf patterns. Drench root zones with Thiophanate Methyl 1.5 grams per liter. Avoid over-irrigation."
        },
        "hi": {
            "disease_name": "वर्टिसिलियम उकठा रोग (Verticillium Wilt)",
            "pathogen": "वर्टिसिलियम डाहली (मृदा जनित कवक)",
            "severity": "मध्यम से उच्च",
            "symptoms": "पत्तियों की नसों के बीच का हिस्सा पीला और बाद में भूरा हो जाता है जिससे बाघ जैसी धारियां (टाइगर-स्ट्राइप) बन जाती हैं। फूल व गूलर गिर जाते हैं।",
            "immediate_action": "रोगग्रस्त पौधों को तुरंत उखाड़कर अलग करें। खेत में अत्यधिक पानी न लगाएं, क्योंकि बहता पानी कवक को पूरे खेत में फैलाता है।",
            "organic_remedy": "ट्राइकोडर्मा विरिडी या स्यूडोमोनास युक्त जैविक खाद जड़ों में दें। सरसों या राई की हरी खाद से बायो-फ्यूमिगेशन करें।",
            "chemical_control": "थियोफिनेट मिथाइल 70% WP (1.5 ग्राम/लीटर) या कार्बेंडाजिम + मैंकोजेब (2 ग्राम/लीटर) का घोल बनाकर जड़ों के पास डालें (drenching)।",
            "preventive_measures": "धान के साथ फसल चक्र अपनाएं (खेत में पानी भरने से कवक नष्ट होता है)। पोटाश खाद की पर्याप्त मात्रा दें।",
            "voice_summary": "कपास में वर्टिसिलियम उकठा रोग पाया गया है। थियोफिनेट मिथाइल 1.5 ग्राम प्रति लीटर से जड़ों को तर करें और ज्यादा पानी न भरें।"
        },
        "mr": {
            "disease_name": "व्हर्टिसिलियम मर रोग (Verticillium Wilt)",
            "pathogen": "व्हर्टिसिलियम डाहली (बुरशी)",
            "severity": "मध्यम ते तीव्र",
            "symptoms": "पानांच्या शिरांमधील भाग पिवळा पडून वाळतो, ज्यामुळे पानावर वाघाच्या अंगावरील पट्ट्यांसारखा (टायगर स्ट्राईप) पॅटर्न दिसतो.",
            "immediate_action": "बाधित झाडे शेतातून काढून टाका. अतिरिक्त पाणी साठू देऊ नका.",
            "organic_remedy": "ट्रायकोडर्मा विरिडी शेणखतात मिसळून झाडांच्या मुळांशी द्यावे.",
            "chemical_control": "थायोफिनेट मिथाईल ७०% WP (१.५ ग्रॅम/लिटर) किंवा कार्बेंडाझिम + मँकोझेब (२ ग्रॅम/लिटर) ने मुळांशी आळवणी करा.",
            "preventive_measures": "भात पिकासोबत फेरपालट करा. नत्राचा अतिवापर टाळा व पोटॅशचे प्रमाण वाढवा.",
            "voice_summary": "कापसावर व्हर्टिसिलियम मर रोग आढळला आहे. थायोफिनेट मिथाईल १.५ ग्रॅम प्रति लिटर पाण्याने मुळाशी आळवणी करा."
        },
        "te": {
            "disease_name": "వర్టిసిలియం ఎండు తెగులు (Verticillium Wilt)",
            "pathogen": "వర్టిసిలియం శిలీంధ్రం",
            "severity": "తీవ్రం",
            "symptoms": "ఆకుల ఈనెల మధ్య పసుపు మరియు గోధుమ చారలు ఏర్పడి పులి చారల వలే కనిపిస్తాయి. కాయలు రాలిపోతాయి.",
            "immediate_action": "రోగానికి గురైన మొక్కలను పీకి వేయండి. పొలంలో నీరు ఎక్కువగా నిల్వ ఉండకుండా చూడండి.",
            "organic_remedy": "ట్రైకోడెర్మా కలిపిన సేంద్రీయ ఎరువును మొక్కల మొదళ్ళలో వేయండి.",
            "chemical_control": "థయోఫనేట్ మిథైల్ 1.5 గ్రా/లీటర్ నీటిలో కలిపి మొక్కల మొదళ్ళలో పోయండి.",
            "preventive_measures": "వరి లేదా జొన్నతో పంట మార్పిడి చేయండి. పొటాష్ వాడకం పెంచండి.",
            "voice_summary": "వర్టిసిలియం తెగులు నివారణకు థయోఫనేట్ మిథైల్ 1.5 గ్రాములు లీటర్ నీటిలో కలిపి మొదళ్ళలో పోయండి."
        },
        "gu": {
            "disease_name": "વર્ટિસિલિયમ સુકારો (Verticillium Wilt)",
            "pathogen": "વર્ટિસિલિયમ ડાહલી (ફૂગ)",
            "severity": "મધ્યમ થી વધુ",
            "symptoms": "પાનની નસો વચ્ચે વાઘના પટ્ટા જેવી (ટાઈગર સ્ટ્રાઈપ) ભાત બને છે. પાન અને ઝીંડવાં ખરી પડે છે.",
            "immediate_action": "રોગિષ્ટ છોડને ખેતરમાંથી દૂર કરો. વધુ પડતું પાણી ન આપવું.",
            "organic_remedy": "ટ્રાઈકોડર્મા યુક્ત છાણીયું ખાતર મૂળમાં આપવું.",
            "chemical_control": "થાયોફેનેટ મિથાઈલ ૭૦% વે.પા. (૧.૫ ગ્રામ/લિટર) અથવા સાફ ફૂગનાશક (૨ ગ્રામ/લિટર) નું ડ્રેન્ચિંગ કરવું.",
            "preventive_measures": "ડાંગર સાથે પાકની ફેરબદલી કરવી. પૂરતા પ્રમાણમાં પોટાશ આપવું.",
            "voice_summary": "કપાસમાં વર્ટિસિલિયમ સુકારા માટે થાયોફેનેટ મિથાઈલ ૧.૫ ગ્રામ પ્રતિ લિટર મૂળમાં રેડો."
        }
    }
}


class AgronomicAdvisoryEngine:
    """Intelligent Agronomic Advisory Engine combining Google Gemini & Verified Offline Knowledge."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None

        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google Gemini Generative AI client ({self.model_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}. Defaulting to offline knowledge base.")
        else:
            logger.info("Operating in standalone/offline mode with expert agronomic knowledge base.")

    def set_api_key(self, api_key: str) -> bool:
        """Dynamically update Gemini API key at runtime."""
        self.api_key = api_key
        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google Gemini Generative AI client ({self.model_name})")
                return True
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client with new key: {e}")
                self.client = None
                return False
        return False

    def generate_advisory(
        self,
        disease_name: str,
        confidence_pct: float,
        language: str = "en",
        field_conditions: Optional[str] = None,
        image_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete agronomic advisory.
        Attempts Gemini Generative AI if configured, otherwise serves offline verified agronomy.
        """
        # Canonicalize disease key
        clean_key = disease_name.strip().replace(" ", "_")
        if clean_key not in OFFLINE_KNOWLEDGE_BASE:
            for k in OFFLINE_KNOWLEDGE_BASE:
                if k.lower() in clean_key.lower():
                    clean_key = k
                    break

        lang = language if language in ("en", "hi", "mr", "te", "gu") else "en"

        # Try Gemini Generative AI first if available
        if self.client:
            try:
                genai_result = self._query_gemini(
                    disease_name=clean_key,
                    confidence_pct=confidence_pct,
                    language=lang,
                    field_conditions=field_conditions,
                    image_path=image_path,
                )
                if genai_result:
                    genai_result["source"] = "Gemini Generative AI"
                    genai_result["language"] = lang
                    return genai_result
            except Exception as e:
                logger.warning(f"Gemini API call failed ({e}). Seamlessly falling back to offline knowledge base.")

        # Fallback to rich offline knowledge base
        fallback = OFFLINE_KNOWLEDGE_BASE.get(clean_key, {}).get(lang)
        if not fallback:
            fallback = OFFLINE_KNOWLEDGE_BASE.get("Healthy", {}).get(lang, {})

        result = dict(fallback)
        result["confidence"] = round(confidence_pct, 2)
        result["source"] = "Verified Agronomic Knowledge Base (Offline Ready)"
        result["language"] = lang
        result["field_conditions_noted"] = field_conditions or "Standard Field Conditions"
        return result

    def _query_gemini(
        self,
        disease_name: str,
        confidence_pct: float,
        language: str,
        field_conditions: Optional[str],
        image_path: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        """Query Gemini using google.genai Interactions API."""
        lang_names = {
            "en": "English",
            "hi": "Hindi",
            "mr": "Marathi",
            "te": "Telugu",
            "gu": "Gujarati",
        }
        target_lang_str = lang_names.get(language, "English")

        prompt = f"""
You are the Chief Cotton Agronomist and Senior Plant Pathologist for an AI Agricultural Advisory System.
A high-resolution edge MobileNetV2 model diagnosed a cotton leaf with:
- Condition: {disease_name.replace('_', ' ')}
- Model Confidence: {confidence_pct:.2f}%
- Farmer's Field Context: {field_conditions or 'Standard semi-arid cotton growing conditions'}

Generate an authoritative, practical, and farmer-friendly advisory in the {target_lang_str} language.
Respond ONLY with a valid JSON object matching this exact schema:
{{
    "disease_name": "{disease_name.replace('_', ' ')} translated/transliterated in {target_lang_str}",
    "pathogen": "Scientific name of pathogen",
    "severity": "Low / Moderate / Severe / Critical",
    "symptoms": "Key visual symptoms on cotton leaves",
    "immediate_action": "Action farmer must take today",
    "organic_remedy": "Specific organic and biological treatments (e.g. neem formulations, bio-agents)",
    "chemical_control": "Specific chemical fungicides/bactericides with exact dosage per liter and per acre",
    "preventive_measures": "Agronomic practices, fertilizer balance, crop rotation to prevent recurrence",
    "voice_summary": "A 2-sentence conversational spoken script in {target_lang_str} suitable for voice read-aloud to an illiterate farmer"
}}
"""
        contents = [prompt]
        if image_path and Path(image_path).exists():
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    contents.append(img.copy())
            except Exception:
                pass

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
        )

        if response and response.text:
            return json.loads(response.text)
        return None


def main():
    parser = argparse.ArgumentParser(description="Cotton Leaf Agronomic Advisory Engine")
    parser.add_argument("--disease", type=str, default="Alternaria_Leaf_Spot", help="Detected disease class")
    parser.add_argument("--confidence", type=float, default=94.5, help="Confidence percentage")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi", "mr", "te", "gu"], help="Language code")
    parser.add_argument("--field", type=str, default=None, help="Optional field weather/soil conditions")
    args = parser.parse_args()

    engine = AgronomicAdvisoryEngine()
    advisory = engine.generate_advisory(
        disease_name=args.disease,
        confidence_pct=args.confidence,
        language=args.lang,
        field_conditions=args.field,
    )

    print("\n" + "=" * 65)
    print("COTTON AGRONOMIC ADVISORY REPORT")
    print("=" * 65)
    print(f"Condition        : {advisory.get('disease_name')}")
    print(f"Pathogen         : {advisory.get('pathogen')}")
    print(f"Severity         : {advisory.get('severity')}")
    print(f"Confidence       : {advisory.get('confidence')}%")
    print(f"Advisory Source  : {advisory.get('source')}")
    print("-" * 65)
    print(f"Symptoms         :\n  {advisory.get('symptoms')}\n")
    print(f"Immediate Action :\n  {advisory.get('immediate_action')}\n")
    print(f"Organic Remedy   :\n  {advisory.get('organic_remedy')}\n")
    print(f"Chemical Control :\n  {advisory.get('chemical_control')}\n")
    print(f"Prevention       :\n  {advisory.get('preventive_measures')}\n")
    print(f"Farmer Voice Script :\n  \"{advisory.get('voice_summary')}\"")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
