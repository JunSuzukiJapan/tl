# Obsługiwane mechanizmy uczenia modeli

Oprócz zaawansowanych operacji tensorowych TensorLanguage (TL) obsługuje uczenie modelu sieci neuronowej. W tym dokumencie wyjaśniono przepływ pracy od definiowania modeli do implementacji pętli szkoleniowej i zapisywania wyszkolonych modeli przy użyciu TL.

## 1. Podstawowe koncepcje szkoleniowe

Szkolenie w TL przebiega w następujących etapach:

1. **Definicja modelu**: Użyj `struct` do zdefiniowania warstw i modeli przechowujących parametry i stan.
2. **Przejście w przód**: Oblicz dane wyjściowe z tensorów wejściowych, aby uzyskać wyniki przewidywań lub logity.
3. **Obliczanie strat i przejście wstecz**: Wywołaj `loss.backward()` na funkcji straty (np. `cross_entropy`), aby obliczyć gradienty każdego parametru.
4. **Optymalizacja**: Wywołaj wbudowane funkcje optymalizacyjne (np. `adam_step`) dla każdego parametru, aby je zaktualizować i zresetuj gradienty za pomocą `Tensor::clear_grads()`.

## 2. Różnice między tensorem a stopniowym tensorem

TL ma dwa podstawowe typy tensorów statycznych używanych do obliczeń numerycznych i uczenia:

- **`Tensor<T, R>`**: Standardowe dane w tablicy wielowymiarowej. Nie śledzi gradientów (historii obliczeń), dzięki czemu jest szybki i oszczędza pamięć. Stosowany jest głównie do **przetwarzania danych podczas wnioskowania** i przechowywania **wewnętrznego stanu optymalizatora (np. pędu i wariancji)**.
- **`GradTensor<T, R>`**: Tensor śledzenia gradientu do treningu. Rejestruje proces obliczeniowy (tworzy wykres obliczeniowy) i wykonuje automatyczne różnicowanie w celu obliczenia gradientów po wywołaniu funkcji „backward()”. Zawsze musisz używać `GradTensor` dla **parametrów (wag, błędów systematycznych itp.), które mają być poznane/zaktualizowane** przez algorytm optymalizacji.

## 3. Definicja i inicjalizacja

Każda warstwa modelu jest zdefiniowana jako „struktura”. Na przykład warstwa liniowa wyszkolona za pomocą optymalizatora Adama musi utrzymywać stan pędu (`m`, `v`) oprócz wag i odchyleń. Do parametrów treningowych przypisujemy „GradTensor”, a do stanu optymalizatora „Tensor”.


__KOD_BLOKU_0__


*Uwaga*: Wywołanie `detach(true)` podczas inicjalizacji parametru jawnie zaznacza ten tensor jako cel obliczeń gradientu.

## 4. Wdrażanie kroku optymalizacji

Dodaj funkcję „step” do każdej warstwy, aby wykonać algorytm optymalizacji (np. Adam) i zaktualizować jego stan. Metoda „krokowa” TL zazwyczaj wykorzystuje niezmienny projekt, który po aktualizacji zwraca nową strukturę.


__KOD_BLOKU_1__


## 5. Pętla treningowa i przejście wstecz

W głównej pętli szkoleniowej oblicz stratę, wywołaj funkcję „backward()”, zaktualizuj model za pomocą funkcji „step”, a następnie wyczyść gradienty.


__KOD_BLOKU_2__


## 6. Zapisywanie modelu (safetensory)

Wyuczone parametry modelu można zapisać w formacie `.safetensors` za pomocą funkcji `Param::save`. Zapisane dane można ponownie wykorzystać do wnioskowania.


__KOD_BLOKU_3__