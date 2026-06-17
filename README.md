Markdown
<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDI3Y2g1cXBlbXF6b3p5NXpndXp5bmt0Z3p5bmt0Z3p5bmt0Z3p5bSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7qE1YN7aBOFPRw8E/giphy.gif" width="100%" height="250px" style="object-fit: cover; border-radius: 10px;" alt="Solar Sentinel Banner">

# 🌌 SolarSentinel AI (v3)

### 🛰️ An Advanced Space-Weather Intelligence Engine for Planetary Infrastructure Defense

<br>

[![Web App](https://img.shields.io/badge/Live_Dashboard-🟢_Online-00C853?style=for-the-badge&logo=react&logoColor=white)](http://localhost:5173/globe-risk)
[![FastAPI](https://img.shields.io/badge/FastAPI_Backend-⚡_Deployed-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://solar-sentinel-v3.onrender.com)
[![Python](https://img.shields.io/badge/Core_Engine-🐍_Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![GitHub Repo](https://img.shields.io/badge/Repository-💾_Source-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Harsh28-raj/solar-sentinel-v3.git)

<br>

> **⚡ Early Warning System:** Processing real-time NASA satellite intelligence to shield global grids from cosmic anomalies.
> Developed by **Harsh Raj** — 2nd Year CS Student | AI/ML Developer (MLCOE)

---

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png" width="100%">

</div>

## 📌 Problem Statement & Theme

**Theme:** `Environmental Sustainability & Space Weather Resilience`

The Sun constantly emits cosmic radiation, but sudden **Solar Flares** and **Heliostorms (Coronal Mass Ejections)** pose a catastrophic threat to Earth's modern infrastructure. A severe solar storm can instantly destroy global power grids, disrupt satellite communications (GPS, 5G, NavIC), and blind military radars.

**SolarSentinel AI** eliminates fragmented, late-stage global alerts by connecting directly to live satellite feeds, automating risk assessment, and introducing deep-space early warnings alongside precise surface target mapping.

---

## 🚀 Core Features & Architecture

<details open>
<summary><b>🌪️ Module 1: Heliostorm Deep-Space Tracker (Click to collapse)</b></summary>
<br>
Processes dynamic deep-space streaming metrics to identify rapid shifts in solar wind density and interplanetary magnetic fields.
<ul>
  <li><b>Capability:</b> Predicts incoming Coronal Mass Ejections (CMEs) and high-energy particle streams.</li>
  <li><b>Impact:</b> Provides a <b>15–30 hour early-warning window</b> for global power grid managers and aerospace agencies to isolate grids and safe-mode satellites.</li>
</ul>
</details>

<br>

<details open>
<summary><b>🎯 Module 2: Solar Flare Prediction & Spatial Localization (Click to collapse)</b></summary>
<br>
A custom Computer Vision pipeline that reads live solar imagery (NASA SOHO/SDO streams) to instantly classify threat thresholds and locate active regions.
<ul>
  <li><b>Risk Engine:</b> Outputs real-time risk profiles (<code>LOW</code>, <code>HIGH</code>, <code>CRITICAL</code>) along with computed <code>real_space_flux</code> metrics.</li>
  <li><b>Spatial Localization:</b> Extracted exact spatial coordinates (<code>x_coordinate_percent</code>, <code>y_coordinate_percent</code>) to render a blinking threat layer directly over the frontend digital Sun mesh.</li>
</ul>
</details>

---

## 📡 Live Deployments & Dynamic Endpoints

<div align="center">
  
| Service | Status | URL / Link |
| :--- | :---: | :--- |
| **Frontend Dashboard** | `🟢 Live` | [http://localhost:5173/globe-risk](http://localhost:5173/globe-risk) |
| **Live API Backend** | `⚡ Operational` | [https://solar-sentinel-v3.onrender.com](https://solar-sentinel-v3.onrender.com) |
| **Interactive Swagger Docs** | `📖 Active` | [https://solar-sentinel-v3.onrender.com/docs](https://solar-sentinel-v3.onrender.com/docs) |

</div>

### 🛠️ Production REST API Endpoints

```bash
# 1. Predict Live NASA Stream
# Processes the current real-time satellite imagery buffer to generate active structural threat scoring.
POST /api/v1/ai/predict-live

# 2. Anomaly Spatial Localization
# Returns exact flare pixel matrix coordinates to draw the dynamic blinking targeting layer.
POST /api/v1/ai/flare-prediction
🗂️ Project Repository Structure
Markdown
solar-sentinel-v3/
│
├── main.py                  # Core FastAPI App & CORS Pipeline Control
├── requirements.txt         # Production Ready Python Dependencies
├── solar_sentinel_cnn.pth   # Pre-trained CV weights for localization
└── [Frontend/React Assets]  # Dynamic Digital Sun Matrix & Globe Risk Mapping
⚙️ Tech Stack
🛡️ Target Audience
🏦 Power Grid Operators: Automates immediate multi-hour isolation protocols to prevent grid burnout.

🛰️ Aerospace & Satellite Agencies (ISRO, SpaceX): Protects active orbital constellation sensors from irreversible ionization.

✈️ Telecom & Aviation Sectors: Minimizes critical navigation routing failures during high-frequency radio blackouts.

👨‍💻 Author
Harsh Raj — @Harsh28-raj

2nd Year CS Student | AI/ML Developer

Machine Learning Centre of Excellence (MLCOE)
