import math
import random


class DeforestationAIEngine:

  def __init__(self, n_samples=500, random_state=42):
    self.n_samples = n_samples
    self.random_state = random_state
    self.training_data = []
    self.class_mapping = {
        0: "Pertanian Skala Kecil",
        1: "Pertanian Skala Industri",
        2: "Pertambangan Tanpa Izin",
        3: "Penebangan Selektif (Logging)",
        4: "Kebakaran Lahan / Bencana Alam",
    }

  def generate_synthetic_dataset(self):
    """Mensimulasikan dataset ekstraksi fitur citra satelit (S1 & S2)."""
    random.seed(self.random_state)
    dataset = []

    for _ in range(self.n_samples):
      features = [random.random() for _ in range(5)]

      if features[4] > 0.75:
        label = 4  
      elif features[1] > 0.6 and features[3] < 0.3:
        label = 2  
      elif features[3] < 0.35 and features[0] < 0.5:
        label = 3  
      elif features[0] > 0.55:
        label = 1  
      else:
        label = 0  
      dataset.append((features, label))
    return dataset

  def train_model(self):
    """Melatih model dengan menyimpan dataset referensi (KNN / Lazy Learning)."""
    print("[INFO] Menyiapkan dataset pelatihan fitur satelit...")
    self.training_data = self.generate_synthetic_dataset()
    print(
        f"[SUKSES] Model siap! Sebanyak {len(self.training_data)} data latih"
        " dimuat."
    )
    return self.training_data

  def _euclidean_distance(self, p1, p2):
    """Menghitung jarak kemiripan antar fitur."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

  def predict_driver(self, features, k=5):
    """Melakukan inferensi real-time untuk mendeteksi pemicu deforestasi."""
    if not self.training_data:
      raise ValueError("Model belum dilatih. Jalankan train_model terlebih dahulu.")

    distances = []
    for train_features, label in self.training_data:
      dist = self._euclidean_distance(features, train_features)
      distances.append((dist, label))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    class_counts = {}
    for _, label in k_nearest:
      class_counts[label] = class_counts.get(label, 0) + 1

    predicted_code = max(class_counts, key=class_counts.get)
    confidence = class_counts[predicted_code] / k

    return {
        "kode_pemicu": int(predicted_code),
        "nama_pemicu": self.class_mapping.get(predicted_code, "Tidak Dikenal"),
        "tingkat_kepercayaan": float(confidence),
    }

if __name__ == "__main__":
  engine = DeforestationAIEngine()
  engine.train_model()

  sample_features = [0.45, 0.70, 0.20, 0.15, 0.85]
  hasil = engine.predict_driver(sample_features)

  print("\n--- HASIL PREDIKSI REAL-TIME ---")
  print(f"Pemicu Deforestasi : {hasil['nama_pemicu']}")
  print(f"Tingkat Keyakinan  : {hasil['tingkat_kepercayaan'] * 100:.1f}%")