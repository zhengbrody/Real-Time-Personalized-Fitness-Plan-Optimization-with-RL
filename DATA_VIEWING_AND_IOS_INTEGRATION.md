# Data Viewing and iOS Integration Guide

## 📊 How to View Results from Your Data

### 1. **Current Data Flow**

```
Raw Data Sources → Preprocessing → Feature Engineering → Model Training → API → Web Interface
     ↓                  ↓                  ↓                  ↓           ↓          ↓
  Apple Watch       Unified Daily      Daily Features    Bandit Model   REST API   Visualizations
  Oura Ring         Aggregates         (45+ features)    (Trained)      Endpoints  & Recommendations
  Training Logs
```

### 2. **Viewing Data at Each Stage**

#### A. **Raw Data**

Your raw data is stored in:
- `data/raw/apple/` - Apple Watch exports
- `data/raw/OURA/` - Oura Ring data
- `data/raw/training_logs/` - Training session logs

**To view:**
```python
import pandas as pd

# View Apple Watch data
apple_data = pd.read_csv('data/raw/apple/export.csv')
print(apple_data.head())

# View Oura Ring data
oura_data = pd.read_csv('data/raw/OURA/oura_daily.csv')
print(oura_data.head())
```

#### B. **Processed Data**

After running preprocessing:
```bash
python src/data_collection/preprocess.py
```

View unified daily data:
```python
import pandas as pd

# Load processed data
daily_data = pd.read_parquet('data/processed/unified_daily.parquet')
print(daily_data.head())
print(f"\nTotal days: {len(daily_data)}")
print(f"Columns: {daily_data.columns.tolist()}")
```

#### C. **Engineered Features**

After feature engineering:
```bash
python src/feature_store/feature_engineering.py
```

View features:
```python
import pandas as pd

# Load feature matrix
features = pd.read_parquet('data/features/daily_features.parquet')
print(features.head())
print(f"\nTotal features: {features.shape[1]}")
print(f"Feature names: {features.columns.tolist()}")
```

#### D. **Model Predictions and Recommendations**

**Via Web Interface** (Easiest):
1. Open `http://localhost:8501`
2. Go to "Get Recommendation" tab
3. Input your body state data
4. View recommendations in real-time
5. Go to "Data Analysis" tab to see trends and history

**Via API** (Programmatic):
```python
import requests

# Get recommendation
response = requests.post(
    'http://localhost:8000/recommend',
    json={
        'user_id': 'user_001',
        'state': {
            'readiness_score': 75,
            'sleep_score': 80,
            'hrv': 50,
            'resting_hr': 60,
            'fatigue': 5,
            'activity_score': 70
        }
    }
)

recommendation = response.json()
print(recommendation)
```

**Via Database**:
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data/fitness.db')

# View training sessions
sessions = pd.read_sql_query("SELECT * FROM training_sessions", conn)
print(sessions)

# View user feedback
feedback = pd.read_sql_query("SELECT * FROM user_feedback", conn)
print(feedback)

# View daily states
states = pd.read_sql_query("SELECT * FROM daily_states", conn)
print(states)

conn.close()
```

### 3. **Creating Custom Visualizations**

Create a Jupyter notebook to analyze your data:

```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load your data
features = pd.read_parquet('data/features/daily_features.parquet')

# Example 1: Sleep Quality vs Readiness Score
fig = px.scatter(
    features,
    x='sleep_score',
    y='readiness_score',
    title='Sleep Quality vs Readiness',
    labels={'sleep_score': 'Sleep Score', 'readiness_score': 'Readiness Score'}
)
fig.show()

# Example 2: HRV Trends Over Time
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=features.index,
    y=features['hrv'],
    mode='lines+markers',
    name='HRV'
))
fig.update_layout(title='HRV Trends', xaxis_title='Date', yaxis_title='HRV (ms)')
fig.show()

# Example 3: Training Load vs Recovery
fig = px.scatter(
    features,
    x='activity_score',
    y='recovery_score',
    color='fatigue_level',
    title='Training Load vs Recovery (colored by fatigue)',
    labels={'activity_score': 'Activity Score', 'recovery_score': 'Recovery Score'}
)
fig.show()
```

---

## 📱 iOS App Integration Plan

### **Question: Can an iOS app automatically sync Oura and Apple Watch data for daily recommendations?**

**Answer: YES!** This is absolutely feasible. Here's how:

### 1. **Technical Architecture**

```
iOS App
   ↓
   ├─→ HealthKit (Apple Watch data) ──┐
   ├─→ Oura Cloud API (Oura Ring data) ┤
   └─→ Manual Input (optional) ────────┘
                                        ↓
                               Data Synchronization
                                        ↓
                               Backend API (FastAPI)
                                        ↓
                               ML Model (Recommendation Engine)
                                        ↓
                               Push Notification
                                        ↓
                               iOS App (Display Recommendation)
```

### 2. **Implementation Steps**

#### **Step 1: Apple Watch Data Sync (via HealthKit)**

HealthKit provides direct access to Apple Watch data on iOS:

```swift
import HealthKit

class HealthDataManager {
    let healthStore = HKHealthStore()

    func requestAuthorization() {
        let typesToRead: Set = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!
        ]

        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            if success {
                self.fetchTodayData()
            }
        }
    }

    func fetchTodayData() {
        // Fetch HRV
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        let today = Calendar.current.startOfDay(for: Date())
        let predicate = HKQuery.predicateForSamples(withStart: today, end: Date(), options: .strictStartDate)

        let query = HKStatisticsQuery(quantityType: hrvType, quantitySamplePredicate: predicate, options: .discreteAverage) { query, result, error in
            if let average = result?.averageQuantity() {
                let hrv = average.doubleValue(for: HKUnit.secondUnit(with: .milli))
                print("HRV: \(hrv) ms")
                // Send to backend
                self.sendToBackend(hrv: hrv)
            }
        }

        healthStore.execute(query)
    }

    func sendToBackend(hrv: Double) {
        // API call to your FastAPI backend
        let url = URL(string: "https://your-api.com/sync-health-data")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "user_id": "user_001",
            "hrv": hrv,
            "date": ISO8601DateFormatter().string(from: Date())
        ]

        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle response
        }.resume()
    }
}
```

#### **Step 2: Oura Ring Data Sync (via Oura Cloud API)**

Oura provides a REST API for accessing ring data:

```swift
import Foundation

class OuraDataManager {
    let baseURL = "https://api.ouraring.com/v2/usercollection"
    var accessToken: String = "YOUR_OURA_ACCESS_TOKEN"

    func fetchDailyReadiness() {
        let url = URL(string: "\(baseURL)/daily_readiness")!
        var request = URLRequest(url: url)
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let data = data {
                let readinessData = try? JSONDecoder().decode(OuraReadiness.self, from: data)
                print("Readiness Score: \(readinessData?.score ?? 0)")
                // Send to backend
                self.sendToBackend(readinessScore: readinessData?.score ?? 0)
            }
        }.resume()
    }

    func fetchDailySleep() {
        let url = URL(string: "\(baseURL)/daily_sleep")!
        var request = URLRequest(url: url)
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let data = data {
                let sleepData = try? JSONDecoder().decode(OuraSleep.self, from: data)
                print("Sleep Score: \(sleepData?.score ?? 0)")
                // Send to backend
            }
        }.resume()
    }
}

struct OuraReadiness: Codable {
    let score: Int
    let contributors: Contributors
}
```

#### **Step 3: Automatic Daily Sync**

Set up background fetch to automatically sync data each morning:

```swift
import UIKit
import BackgroundTasks

class AppDelegate: UIResponder, UIApplicationDelegate {

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {

        // Register background task
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.yourapp.dailysync",
            using: nil
        ) { task in
            self.handleDailySync(task: task as! BGAppRefreshTask)
        }

        return true
    }

    func scheduleAppRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: "com.yourapp.dailysync")

        // Schedule for 7 AM tomorrow
        var components = Calendar.current.dateComponents([.year, .month, .day], from: Date())
        components.hour = 7
        components.minute = 0
        request.earliestBeginDate = Calendar.current.date(from: components)

        try? BGTaskScheduler.shared.submit(request)
    }

    func handleDailySync(task: BGAppRefreshTask) {
        // Fetch data from HealthKit and Oura
        let healthManager = HealthDataManager()
        let ouraManager = OuraDataManager()

        healthManager.fetchTodayData()
        ouraManager.fetchDailyReadiness()
        ouraManager.fetchDailySleep()

        // Get recommendation from backend
        fetchRecommendation { recommendation in
            // Send push notification
            self.sendNotification(recommendation: recommendation)
            task.setTaskCompleted(success: true)
        }

        // Schedule next sync
        scheduleAppRefresh()
    }

    func fetchRecommendation(completion: @escaping (Recommendation) -> Void) {
        let url = URL(string: "https://your-api.com/recommend")!
        // API call to get recommendation
    }

    func sendNotification(recommendation: Recommendation) {
        let content = UNMutableNotificationContent()
        content.title = "Today's Training Plan"
        content.body = "Your recommended workout: \(recommendation.workoutType)"
        content.sound = .default

        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }
}
```

### 3. **User Experience Flow**

```
Morning (7:00 AM):
1. iOS app automatically fetches data from:
   - Apple Watch (via HealthKit): HRV, resting HR, sleep, activity
   - Oura Ring (via API): readiness score, sleep quality

2. App sends data to backend API

3. Backend ML model generates recommendation

4. iOS app receives recommendation

5. Push notification sent to user:
   "Good morning! Based on your sleep (80/100) and readiness (75/100),
    today's recommended workout: Moderate Cardio, 30 minutes"

User opens app:
6. View detailed recommendation with rationale

7. Chat with AI coach for questions

8. After workout, submit feedback (RPE, mood, satisfaction)

9. Data stored for continuous learning
```

### 4. **Backend API Enhancements Needed**

Add these endpoints to your FastAPI backend:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class HealthData(BaseModel):
    user_id: str
    date: datetime
    hrv: float
    resting_hr: int
    sleep_score: int
    readiness_score: int
    activity_score: int

class DailyRecommendation(BaseModel):
    user_id: str
    workout_type: str
    intensity: str
    duration_minutes: int
    rationale: str

@app.post("/sync-health-data")
async def sync_health_data(data: HealthData):
    """
    Receive health data from iOS app and store in database
    """
    # Store in database
    save_to_database(data)

    # Generate recommendation
    recommendation = generate_recommendation(data)

    return {
        "status": "success",
        "recommendation": recommendation
    }

@app.get("/daily-recommendation/{user_id}")
async def get_daily_recommendation(user_id: str):
    """
    Get today's recommendation for user
    """
    # Fetch latest health data
    health_data = get_latest_health_data(user_id)

    # Generate recommendation
    recommendation = generate_recommendation(health_data)

    return recommendation

@app.post("/register-device")
async def register_device(user_id: str, device_token: str):
    """
    Register iOS device for push notifications
    """
    save_device_token(user_id, device_token)
    return {"status": "registered"}
```

### 5. **Push Notification Service**

Use Apple Push Notification Service (APNs):

```python
from aioapns import APNs, NotificationRequest

async def send_push_notification(user_id: str, recommendation: dict):
    """
    Send push notification to iOS device
    """
    device_token = get_device_token(user_id)

    apns = APNs(
        key='path/to/key.p8',
        key_id='YOUR_KEY_ID',
        team_id='YOUR_TEAM_ID',
        topic='com.yourapp.fitnesscoach'
    )

    notification = NotificationRequest(
        device_token=device_token,
        message={
            "aps": {
                "alert": {
                    "title": "Today's Training Plan",
                    "body": f"{recommendation['workout_type']} - {recommendation['duration_minutes']} minutes"
                },
                "sound": "default",
                "badge": 1
            },
            "recommendation": recommendation
        }
    )

    await apns.send_notification(notification)
```

### 6. **Privacy and Permissions**

iOS app needs to request:
- ✅ HealthKit access (Apple Watch data)
- ✅ Network access (API calls)
- ✅ Push notifications
- ✅ Background app refresh

**Info.plist entries:**
```xml
<key>NSHealthShareUsageDescription</key>
<string>We need access to your health data to provide personalized training recommendations</string>

<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>processing</string>
    <string>remote-notification</string>
</array>
```

### 7. **Offline Support**

Cache recommendations for offline access:

```swift
class RecommendationCache {
    static let shared = RecommendationCache()

    func saveRecommendation(_ recommendation: Recommendation) {
        UserDefaults.standard.set(try? JSONEncoder().encode(recommendation), forKey: "lastRecommendation")
    }

    func getLastRecommendation() -> Recommendation? {
        guard let data = UserDefaults.standard.data(forKey: "lastRecommendation") else { return nil }
        return try? JSONDecoder().decode(Recommendation.self, from: data)
    }
}
```

---

## 📊 Summary

### **Data Viewing Options:**

1. **Web Interface** (Easiest) - `http://localhost:8501`
   - Real-time visualizations
   - Historical trends
   - Export to CSV

2. **API** (Programmatic) - `http://localhost:8000`
   - Direct JSON responses
   - Integration with other tools

3. **Database** (Raw Data) - `data/fitness.db`
   - SQL queries
   - Complete data access

4. **Parquet Files** (Data Science) - `data/features/*.parquet`
   - Pandas analysis
   - Custom visualizations

### **iOS Integration:**

✅ **Fully Feasible**
- HealthKit provides Apple Watch data
- Oura Cloud API provides ring data
- Background fetch enables automatic sync
- Push notifications deliver recommendations
- AI coach available in-app

**Timeline Estimate:**
- Basic iOS app with sync: 2-3 weeks
- Full-featured app with AI coach: 4-6 weeks
- App Store submission: +1-2 weeks

**Next Steps:**
1. Set up FastAPI endpoints for mobile
2. Develop iOS app UI/UX
3. Implement HealthKit integration
4. Add Oura API integration
5. Set up push notifications
6. Beta testing with TestFlight

---

**Questions? Let me know if you need help with any specific aspect!**
