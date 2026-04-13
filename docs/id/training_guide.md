# Mekanisme Pelatihan Model yang Didukung

TensorLanguage (TL) mendukung pelatihan model jaringan saraf selain operasi tensor yang kuat. Dokumen ini menjelaskan alur kerja mulai dari menentukan model hingga mengimplementasikan loop pelatihan dan menyimpan model terlatih menggunakan TL.

## 1. Konsep Dasar Pelatihan

Pelatihan TL dilakukan dengan langkah-langkah berikut:

1. **Definisi Model**: Gunakan `struct` untuk menentukan lapisan dan model yang menyimpan parameter dan status.
2. **Forward Pass**: Menghitung keluaran dari tensor masukan untuk mendapatkan skor prediksi atau logit.
3. **Loss Computation dan Backward Pass**: Panggil `loss.backward()` pada fungsi kerugian (misalnya, `cross_entropy`) untuk menghitung gradien setiap parameter.
4. **Optimasi**: Panggil fungsi pengoptimalan bawaan (misalnya, `adam_step`) pada setiap parameter untuk memperbaruinya, dan setel ulang gradien dengan `Tensor::clear_grads()`.

## 2. Perbedaan Tensor dan GradTensor

TL memiliki dua tipe tensor statis utama yang digunakan untuk komputasi dan pelatihan numerik:

- **`Tensor<T, R>`**: Data array multidimensi standar. Itu tidak melacak gradien (riwayat komputasi), sehingga cepat dan hemat memori. Hal ini terutama digunakan untuk **pemrosesan data selama inferensi** dan menyimpan **keadaan internal pengoptimal (misalnya momentum dan varians)**.
- **`GradTensor<T, R>`**: Tensor pelacakan gradien untuk pelatihan. Ini mencatat proses komputasi (membuat grafik komputasi) dan melakukan diferensiasi otomatis untuk menghitung gradien ketika `mundur()` dipanggil. Anda harus selalu menggunakan `GradTensor` untuk **parameter (bobot, bias, dll.) agar dipelajari/diperbarui** oleh algoritme pengoptimalan.

## 3. Definisi dan Inisialisasi

Setiap lapisan model didefinisikan sebagai `struct`. Misalnya, lapisan Linear yang dilatih dengan pengoptimal Adam perlu mempertahankan status momentum (`m`, `v`) selain bobot dan bias. Kami menetapkan `GradTensor` ke parameter pelatihan, dan `Tensor` ke status pengoptimal.


__KODE_BLOK_0__


*Catatan*: Memanggil `detach(true)` selama inisialisasi parameter secara eksplisit menandai tensor ini sebagai target komputasi gradien.

## 4. Menerapkan Langkah Optimasi

Tambahkan fungsi `langkah` ke setiap lapisan untuk menjalankan algoritme pengoptimalan (misalnya Adam) dan memperbarui statusnya. Metode `langkah` TL biasanya menggunakan desain yang tidak dapat diubah yang mengembalikan struktur baru setelah pembaruan.


__KODE_BLOK_1__


## 5. Latihan Loop dan Mundur

Di loop pelatihan utama, hitung kerugiannya, panggil `backward()`, perbarui model melalui `langkah`, lalu hapus gradiennya.


__KODE_BLOK_2__


## 6. Menyimpan Model (Safetensor)

Parameter model yang dipelajari dapat disimpan dalam format `.safetensors` menggunakan fungsi `Param::save`. Data yang disimpan dapat digunakan kembali untuk inferensi.


__KODE_BLOK_3__