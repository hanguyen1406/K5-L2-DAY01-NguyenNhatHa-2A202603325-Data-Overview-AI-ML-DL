# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 
Ultralytics `8.4.145`
Python: 3.13.16
PyTorch: 2.11.0+cu130

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

1. Record hạng 1 của `traffic`: 
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
}
2. Record nói về **cả bức ảnh**, vì classification gán một lớp hoặc ranking cho ảnh đó, không vẽ box hoặc mask cho từng vật thể.
3. Class list đi kèm taxonomy/checkpoint (`ImageNet-1K`), không phải model tự nghĩ ra tên lớp mới.
4. `class_id` là mã ổn định để máy xử lý, `class_name` dễ đọc cho người, còn `taxonomy_name` cho biết bộ quy ước mà mã/tên đó thuộc về. Chỉ ghi tên lớp có thể gây nhầm lẫn.
5. Guideline phải nêu rõ ảnh được gán theo chủ thể chính nào, tỷ lệ diện tích, thứ tự ưu tiên, quy tắc loại trừ và quy ước khi 2 vật thể đều thỏa mãn như nhau.
6. `score` chỉ là mức model tin vào prediction. Nó không phải nhãn đúng do người gán và không chứng minh ảnh thực sự thuộc lớp đó; ground truth phải dựa trên guideline và việc xem ảnh.

## 2. Phát hiện vật thể

1. Chọn record
{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  },

2. Box bắt đầu tại góc trên-trái khoảng `(385.33, 69.24)` pixel và kết thúc tại góc dưới-phải khoảng `(498.92, 348.92)` pixel. Box rộng `113.58` pixel và cao `279.68` pixel.
3. Chọn hai ngưỡng được lưu/kiểm chứng từ output:
   - `0.35`: 11 prediction cho `kitchen` (đây là ngưỡng của JSON).
   - `0.60`: 6 prediction, là các record có `score >= 0.60`.
   Hạ ngưỡng thường giữ thêm các hộp điểm thấp; ngưỡng chỉ lọc prediction, không phải quy tắc bỏ vật thể khỏi ground truth.
4. Khi giữ nhiều hộp hơn, reviewer phải xem thêm false positive, hộp trùng/lệch, lớp sai và các hộp nhỏ hoặc bị che; sau đó duyệt, sửa hoặc loại theo guideline.
5. Quy tắc box: vẽ một hình chữ nhật bao trọn phần nhìn thấy của **một** vật thể, chạm sát biên vật thể nhưng không lấy phần nền; nếu vật chạm/cắt mép hoặc bị che thì áp dụng đúng quy tắc visibility đã thống nhất và đánh dấu ca không rõ để review.
6. Người ở bên phải bị che một phần bởi các vật dụng/đồ bếp phía sau. Đây là ca nên **escalation** vì khó xác định biên phần bị che một cách nhất quán; không nên tự bịa quy tắc từ một ảnh đơn lẻ. Lab Coach cần chốt quy tắc occlusion áp dụng cho cả tập dữ liệu.

## 3. Phân đoạn theo từng đối tượng

1. Record đã chọn: `instance_id=kitchen-001`; `class_name=person`; `score=0.899318`; `polygon_point_count=348`; 5 điểm đầu của `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`.
2. Box chỉ cho biết hình chữ nhật bao quanh; polygon cho biết biên dạng thực tế của vật thể, gồm phần lõm, chân tay và các vùng không thuộc vật thể trong box.
3. `instance_id` định danh riêng một đối tượng trong output của ảnh, ví dụ `kitchen-001`; nó không cho biết lớp (`class_id`) và cũng không theo dõi cùng đối tượng qua nhiều frame/ảnh như tracking ID.
4. Quy tắc vẽ mask: đi theo biên nhìn thấy của đúng instance, bao gồm các phần thuộc vật thể và loại phần nền; không nối sang vật khác khi hai vật chạm nhau. Vùng bị che xử lý theo quy tắc occlusion đã được guideline phê duyệt.
5. Biên người với tạp dề, tay/chân và các vật phía sau có chỗ khó tách; bàn và các đồ vật trên bàn cũng chạm/chồng lấn nhau. Mình dùng **escalation** cho ca biên không chắc chắn, vì cần một quyết định chung trước khi áp dụng cho các ảnh khác.

## 4. Vòng đời và kiểm tra chất lượng

Vòng đời: `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một `class_id`/`class_name` cho toàn ảnh, theo taxonomy; có thể có quy tắc single-label | `traffic` có top-1 `cab` chỉ `0.510915`, trong khi ảnh có nhiều phương tiện; prediction không phải ground truth | Đọc guideline, chọn lớp phù hợp với chủ đề/ưu tiên đã quy định, không dùng score làm nhãn | Kiểm tra lớp, taxonomy, trường hợp nhiều chủ đề và tính nhất quán giữa annotator |
| Phát hiện vật thể | Mỗi instance: lớp + `bbox_xyxy` theo pixel `[x_min,y_min,x_max,y_max]` | `kitchen` có nhiều vật chạm/che; một số box sát mép ảnh, ví dụ box `person` bắt đầu gần `x=0` | Vẽ box cho từng vật theo quy tắc visibility/occlusion và gắn lớp đúng | Kiểm tra bỏ sót, false positive, lớp, biên box, box trùng và vật bị che/cắt |
| Instance segmentation | Mỗi instance: lớp + polygon các điểm pixel, có thể kèm box | Polygon người có `348` điểm; biên người với tạp dề/vật phía sau và đồ trên bàn khó tách | Vẽ polygon theo biên instance, không nhập các vùng nền hoặc instance khác | Kiểm tra biên mask, tách instance, lỗ/đoạn sai, class và các vùng chạm/che |

## 5. An toàn dữ liệu

1. Chỉ dùng ảnh COCO công khai đã được phép cho bài lab, không đưa ảnh cá nhân, dữ liệu nội bộ hoặc thông tin nhạy cảm lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, mình sẽ dừng xử lý và báo Lab Coach/giảng viên để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` đã điền
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS` (không có log validation trong repository để xác nhận)
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp link repository trên LMS
