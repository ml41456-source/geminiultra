# ===========================================================
# profiles_v11.py
# Dynamic Profile Configurations for StoryGrid v3.4
# V11.1: Attribute Negation definitions (for Persistence Logic)
# ===========================================================

# Define the default profile (Pixar 3D, Vietnamese)
PIXAR_3D_VI_PROFILE = {
    "language": "vi",
    "genre": "pixar_3d_animation",
    "StoryGrid_Version": "3.4", # V11 Update
    "parsing_rules": {
        "default_passive_action": "quan sát",
        "default_passive_emotion": None, 

        # V11.1: Attribute Negations (Actions that remove specific Appearance attributes)
        "attribute_negations": {
            # Action: [List of Appearance attributes it removes]
            "phủi đất": ["dính đầy đất", "lấm lem"],
            "lau khô": ["ướt sũng"],
        },

        "scene_header_patterns": [
            r"^\s*(?:[\-\*\•#]+\s*)?(?:CẢNH|Cảnh)\s+(\d+|MỞ\s*ĐẦU|KẾT)\s*[:\-]?\s+.*$",
        ],
        "structure_labels": {
            "Bối cảnh": "setting", "Hành động": "action", "Sự xuất hiện": "arrival",
            "Cao trào mở đầu": "climax", "Cao trào": "climax", "Kết": "conclusion"
        },
        "character_blacklist": [
            "Cảnh", "Scene", "VÚT", "ĐÂY", "Màu", "Hả", "Ồ",
            "Sẵn sàng", "Bắt đầu", "CẢNH MỞ ĐẦU",
            "CHÍNH XÁC", "KHỔNG LỒ", "MÁT LẠNH", "TUYỆT VỜI", "SỨC MẠNH CỦA MÀU SẮC",
        ],
        "group_agents": ["Ba bạn nhỏ", "Cả lớp", "Các bạn"],

        "invalid_name_suffixes": ["đang", "đặt", "chạy", "ngồi", "bước", "kéo", "nhìn", "không", "chậm rãi", "khoan thai", "bước tới"],
        "invalid_name_prefixes": ["ăn", "bảo", "chọn", "giúp", "nhưng", "phải", "nếu", "con đoán"],
        "role_self_teacher": ["thầy","cô"],
        
        "negation_keywords": ["không", "chưa", "đừng", "ngừng"],
        
        "upper_chars": "A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ",
        "lower_chars": "a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ",
    },
    "lexicons": {
        "appearance_phrases": [
            # Inherent/Static
            "Khăn quàng xanh", "Nơ hồng", "Xinh đẹp",
            "khăn", "nơ", "mũ", "kính", "áo", "quàng",
            "xanh", "đỏ", "vàng", "hồng", "tím", "cam",
            "xinh", "đẹp",
            # Dynamic/Temporary Appearance
            "dính đầy đất", "lấm lem", "ướt sũng", "bay phấp phới", "rực rỡ",
            "mắt tròn xoe"
        ],
        "emotion_phrases": [
            "hào hứng", "bất ngờ", "quyết tâm", "tức giận", "ái ngại", 
            "long lanh", "suy nghĩ", "tự tin", "háo hức", "mắt sáng rỡ",
            "bí ẩn", "vội vàng", "dũng cảm"
        ],
        "action_phrases": [
            "mỉm cười", "cười hiền", "cười lớn", "khoan thai", "nhỏ nhẹ", "gật gù", "lí nhí", "chậm rãi",
            "hít hít mũi", "gãi bụng", "chỉnh lại nơ", "dậm dậm chân", "thở hổn hển",
            "chạy vòng quanh", "ngáp ngắn ngáp dài", "soi mình", "bước tới", "đặt chiếc giỏ",
            "xúm lại", "kéo tấm khăn", "nhún nhảy", "vươn vai", "chỉ vào",
            "lao đi", "lật đật chạy", "đi bộ", "quan sát", "vồ ngay lấy", "vật lộn ôm",
            "cầm lên", "ném xuống đất", "quay lưng lại", "khoanh tay", "bước đến", "nhặt lên",
            "phủi đất", "cầm", "nhìn", "nảy mầm", "ngước nhìn", "đứng dậy", "hít một hơi",
            "dừng lại", "ôm chầm lấy", "trang trí", "quây quần", "chia cho", "cắn thử",
            "giơ tay", "chạy đến giỏ", "nhắm mắt", "vỗ vỗ vào mai", "cùng cười", "chạy lại", "vơ ngay lấy",
            "bước đến", "kéo ra", "nói rất nhanh"
        ],
        
        "props_list": ["giỏ mây","cà chua","cà rốt","súp lơ","bảng màu","hoa","sách","bút",
              "khăn lanh","cà tím","ớt chuông vàng","củ dền","dưa chuột","bông cải xanh",
              "táo","chuối","chanh","hoa cúc ánh dương","phiến đá","búp măng", "tấm khăn lanh màu be",
              "quả cà chua", "củ cà rốt", "quả táo", "quả ớt chuông vàng", "rau củ", "chiếc giỏ",
              "vũng nước", "gốc cây cổ thụ"],
        "tone_map": [("ấm áp","warm"),("tò mò","curious"),("nhẹ nhàng","gentle"),("hào hứng","excited"), ("bí ẩn", "mysterious")]
    },
    "cinematic_instructions": {
        "camera_moves": {
            "slow-motion": "slow_motion",
            "chậm lại": "slow_motion",
            "hài hước": "comedic_timing",
        },
        "vfx_sfx": {
            "bụi bay mù mịt": "dust_cloud_vfx",
            "lấp lánh": "sparkle_vfx",
        },
        "meta_types": {
            r"đoạn phim ngắn \d+D hiện lên": "insert_2d_animation",
            r"hình ảnh tưởng tượng": "imagination_sequence",
            r"phim kết thúc với": "end_scene_visual",
            r"Lồng tiếng": "voice_over",
            r"Trở về thực tại": "return_to_reality",
        }
    }
}

# Registry and Loader Function
PROFILES_REGISTRY = {
    "pixar_3d_vi": PIXAR_3D_VI_PROFILE
}

def load_profile(profile_name: str):
    if profile_name in PROFILES_REGISTRY:
        return PROFILES_REGISTRY[profile_name]
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy profile '{profile_name}'. Sử dụng mặc định 'pixar_3d_vi'.")
        return PIXAR_3D_VI_PROFILE