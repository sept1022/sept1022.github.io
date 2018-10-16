---
title: "[Latex] Equations"
categories:
  - latex
tags:
  - latex
  - equation
sitemap: false
---

지난 포스트의 마지막 항목([6. Apply Latex]({% post_url 2018-10-12-jekyll-install-and-use-latex %}))에서 
수식을 이용하기 위한 환경을 갖추었으니, Latex로 표현할 수 있는 수식을 정리해 보겠습니다.

### 1. Inserting Equation

Latex는 `$$ ... $$`와 같이 dollar로 구분해 줍니다. 
예를 들어, `$$1+2=3$$`는 다음의 수식을 표현합니다.

$$1+2=3$$

또한 문장 내에 삽입도 가능합니다. `Equation: $$1+2=3$$ 입니다.`는 다음과 같이 표현됩니다.
Equation: $$1+2=3$$ 입니다.

### 2. Powers & Indices

Power는 **hat(^)**으로 표현합니다. 예를 들어 `$$n^2$$`는 $$n^2$$를 의미합니다.<br>
Indices는 **underscore(_)**로 표현합니다. `$$2_a$$`는 $$2_a$$를 의미합니다.

### 3. Fraction

Fraction은 `\frac{numerator}{denominator}`의 형식으로 표현합니다.<br>
`$$\frac{a}{3}$$`은 $$\frac{a}{3}$$를 표현하며,
다음과 같이 중첩된 형태로도 사용이 가능합니다.

`$$\frac{y}{\frac{3}{x}+b}$$` produces: $$\frac{y}{\frac{3}{x}+b}$$

