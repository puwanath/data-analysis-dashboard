# Advanced Data Analysis Dashboard

ระบบวิเคราะห์ข้อมูลอัจฉริยะที่รองรับการวิเคราะห์ข้อมูลหลากหลายรูปแบบ พร้อมระบบ AI สำหรับการวิเคราะห์เชิงลึก

## คุณสมบัติหลัก

### การจัดการข้อมูล
- รองรับการอัพโหลดไฟล์หลายไฟล์ (CSV, Excel)
- วิเคราะห์ความสัมพันธ์ระหว่างชุดข้อมูล
- ระบบ Merge และ Group ข้อมูลอัตโนมัติ
- ตรวจสอบคุณภาพข้อมูล

### การวิเคราะห์ข้อมูล
- ระบบตรวจจับ Anomaly อัตโนมัติ
- การวิเคราะห์แนวโน้มและการพยากรณ์
- การประมวลผลแบบขนาน
- การวิเคราะห์ด้วย Local LLMs

### การแสดงผล
- สร้าง Dashboard แบบปรับแต่งได้
- Visualization หลากหลายรูปแบบ
- ระบบบันทึกและโหลด Dashboard
- ระบบสร้างรายงานอัตโนมัติ

### ระบบเสริม
- ระบบรักษาความปลอดภัยขั้นสูง
- ระบบ Cache
- รองรับหลายภาษา
- เชื่อมต่อกับแหล่งข้อมูลภายนอก

## การติดตั้ง

1. Clone repository:
```bash
git clone https://github.com/puwanath/data-analysis-dashboard.git
cd data-analysis-dashboard
```

2. สร้าง virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

4. ตั้งค่า environment variables:
```bash
cp .env.example .env
# แก้ไขไฟล์ .env ตามความเหมาะสม
```

5. รัน application:
```bash
streamlit run app.py
```

## การใช้งาน

### 1. อัพโหลดข้อมูล
- เข้าไปที่หน้า "Data Upload"
- อัพโหลดไฟล์ CSV หรือ Excel
- ระบบจะวิเคราะห์ความสัมพันธ์ระหว่างข้อมูลอัตโนมัติ

### 2. วิเคราะห์ข้อมูล
- ใช้ Data Explorer เพื่อดูภาพรวมข้อมูล
- สร้าง Visualization ต่างๆ
- ใช้ AI Analysis สำหรับการวิเคราะห์เชิงลึก

### 3. สร้าง Dashboard
- เลือก Visualization ที่ต้องการ
- จัดวาง Layout
- บันทึก Dashboard สำหรับใช้งานในอนาคต

### 4. สร้างรายงาน
- เลือกรูปแบบรายงาน
- ปรับแต่งเนื้อหา
- Export เป็น PDF หรือ HTML

## การกำหนดค่า

### Security Settings
```yaml
# config/security.json
{
    "password_policy": {
        "min_length": 8,
        "require_uppercase": true,
        "require_lowercase": true,
        "require_numbers": true,
        "require_special": true
    }
}
```

### Cache Settings
```yaml
# config/cache.yaml
{
    "enabled": true,
    "ttl": 3600,
    "max_entries": 1000
}
```

## การพัฒนาเพิ่มเติม

1. สร้าง Custom Visualization:
```python
# src/visualization.py
class CustomVisualizer(Visualizer):
    def create_custom_chart(self, data):
        # Implementation
        pass
```

2. เพิ่มภาษาใหม่:
```yaml
# translations/new_lang.yaml
general:
  welcome: "Welcome message"
  # ...
```

## Contributing

1. Fork repository
2. สร้าง feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.