---
title: "[OSX] Installing Jekyll with minimal mistake and Using Latex"
categories:
  - Jekyll
tags:
  - jekyll
  - latex
  - minimal mistake
  - osx
sitemap: true
---

## 1. Install Command line tool
Jekyll이 사용하는 native extension을 컴파일하기 위해서 command-line tool을 설치합니다.

{% highlight bash %}
xcode-select --install
{% endhighlight %}

## 2. Install jekyll and bundler
{% highlight bash %}
gem install bundler jekyll
{% endhighlight %}

## 3. Create and Clone New Repository
- github에 블로그로 사용할 저장소를 생성합니다. 저장소 이름은 `계정명.github.io`로 합니다.
- local에서 작업하기 위해 해당 디렉토리를 clone 합니다.

{% highlight bash %}
git clone https://github.com/your_id/your_id.github.io.git
{% endhighlight %} 	

## 4. Apply [minimal mistakes][minimal-mistakes] Theme
### Copy Theme
`clone or download`탭을 통해 다운로드 받은 파일을 압축 해제하고, 아래의 파일들을 지우고 나머지 파일들은 위에서 클론해둔 디렉토리에 복사합니다.
	
	- .editorconfig 
	- .gitattributes
	- .github 
	- /docs 
	- /test 
	- CHANGELOG.md
	- minimal-mistakes-jekyll.gemspec
	- README.md
	- screenshot-layouts.png
	- screenshot.png

### Install Plugin
minimal-mistakes가 필요로 하는 plugin을 Gemfile에 정의합니다.

```bash
source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins

group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-sitemap"
  gem "jekyll-gist"
  gem "jekyll-feed"
  gem "jemoji"
  gem "jekyll-algolia"
  gem "jekyll-include-cache"
end
```

Gemfile에 정의된 plugin을 설치하기 위해 다음의 명령어를 수행합니다.

{% highlight bash %}
bundle install
{% endhighlight %}
 	
다음과 같은 메시지와 함께 정상적으로 종료되면 성공입니다.

{% highlight bash %}
Bundle complete! 8 Gemfile dependencies, 94 gems now installed.
Use `bundle info [gemname]` to see where a bundled gem is installed.
{% endhighlight %}

### Run Server
적용한 테마가 정상적으로 동작하는지 확인하겠습니다.

{% highlight bash %}
bundle exec jekyll serve
{% endhighlight %}

빈 페이지이지만, 다음과 같이 정상적으로 적용되었음을 확인할 수 있습니다.

|![Image Alt 텍스트](/assets/img/minimal-mistakes-initialized.png)|
|:--:|
| 초기 실행 화면 |

페이지 좌측에 보이는 `Site Title`, `Your Name`, `Some Where`는 `_config.yml`에 정의된 내용들을 수정하면 쉽게 적용할 수 있습니다.
 
## 5. Posting
저와 같은 방법으로 설치하면, `_posts` 디렉토리가 없으니, `mkdir _posts` 명령어로 디렉토리를 생성하고, 
 `yyyy-mm-dd-posting_name.md`의 형식에 따라 `2016-02-24-welcome-to-jekyll.md`의 파일명으로 다음의 내용을 입력한 후 저장합니다.

{% highlight bash linenos %}
---
title:  "Welcome to Jekyll!"
header:
  teaser: "https://farm5.staticflickr.com/4076/4940499208_b79b77fb0a_z.jpg"
categories: 
  - Jekyll
tags:
  - update
---

You'll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.
To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.
Jekyll also offers powerful support for code snippets:

```ruby
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
```

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
{% endhighlight %}

Posting이 제대로 되었는지 127.0.0.1:4000을 통해 확인합시다.

## 6. Apply Latex
Github Markdown은 Latex를 지원하지 않으므로 수식이 필요한 Posting에 아래의 내용처럼 MathJax를 적용하면, 수식을 사용할 수 있습니다. (필자는 `_include/head.html`에 적용함)  

{% highlight bash %}
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
{% endhighlight %}
 
수식이 잘 작동하는지 확인하기 위해 Posting에 다음의 내용을 추가합니다.
{% highlight bash %}
$$
a^2 + b^2 = c^2
$$ 
{% endhighlight %}
 
수식이 잘 적용되었는지 확인해 보겠습니다.
 
|![applied_latex](/assets/img/applied_latex.png)|
|:--:|
| 수식 적용 |

잘 적용되었네요. 이제 블로그를 운영할 수 있는 준비가 되었습니다.

[minimal-mistakes]: https://github.com/mmistakes/minimal-mistakes

