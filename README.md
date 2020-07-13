# DeZeroCpp
ゼロから作るDeep Learning ❸ をC++で実装する。

## 概要
「[ゼロから作るDeep Learning ❸](https://www.oreilly.co.jp/books/9784873119069/)」を参考にしながら、ディープラーニング用フレームワーク DeZero を C++ で実装しながら学習する。  

## はじめに
目的は、「ゼロから作るDeep Learning ❸」を再読しながら理解を深めることと、C++ の復習。  

### 環境構成
DeZero で使用している Python 用のライブラリや外部ツールの全てを C++ で用意することは難しいかもしれないが、取り合えず以下の構成で始めてみる。  
Windows 上で Visual Studio 2017 以降を使用する。  

| DeZero     | DeZeroCpp                                                                                       |
|------------|-------------------------------------------------------------------------------------------------|
| Python 3   | C++ (MSVC)                                                                                      |
| NumPy      | [NumCpp](https://github.com/dpilger26/NumCpp) (v2.00) + [boost](https://www.boost.org/) (v1.73) |
| Matplotlib | 未定。                                                                                          |
| CuPy       | 未定。多分 GUI 対応は不可能。                                                                   |
| Pillow     | 未定                                                                                            |
| Graphviz   | Graphviz                                                                                        |

- NumCpp には boost も必要。ただし、libファイルは不要なのでダウンロードしてらビルドせずに配置するだけで良い。
    - プロジェクトのプロパティで、「追加のインクルードディレクトリ」に 2 つのライブラリのパスを追加しておく。
- Graphviz はテキストインタフェースのツールなのでそのまま使える。

### 方針
- NumCpp や boost のように、ヘッダオンリーのライブラリを目指す。
    - step07 で破綻した。最終的に、h/cpp にクラスの宣言と定義を分けることにする。
- 復習が目的なので、気づいた点はなるべく細かくメモする。
- 最低限ステップごとにコミットする。ビルドは通る状態とする。


## 開発記録
### ステップ 1：箱としての変数
- Variable クラス
    - `__init__` は C++ ではコンストラクタになる。初期化代入子を使ってメンバを初期化。
    - 今後、継承することになるため、仮想デストラクタを用意しておく。
- NumCpp
    - NumPy は n 次元のテンソルを扱えるが、NumCpp は 2 次元の行列固定。3 次元以上のテンソルが CNN のあたりで出てくるが、まだまだ先のことだし、NumPy ライクに使える他のライブラリが見つからないので、このまま進める。
    - スカラーやベクトルも 2 次元で保持されてしまうようなので、適宜変換が必要かもしれない。
    - NumCpp の warning C4819 (文字コード関係) が 3 つ出るが、とりあえず保留にして進む。
- その他
    - 命名規則は C++ の標準的なスタイルに従う。メンバ変数のスタイルは色々あるようだけど、特に接頭辞や接尾辞は付けないスタイルとする。
    - 名前空間は `dz` としてまとめる。
    - よく使う型は using でエイリアスを作っておく。

### ステップ 2：変数を生み出す関数
- Function クラス
    - オーバーライド必須の関数は、Python では `NoImplementedError` 例外を投げているが、C++ では純粋仮想関数とした。
- step02
    - Python の `type(y)` は、C++ では `typeid(y).name()` に置き換えた。
- その他
    - 書籍のステップに合わせて、stepXX という関数を作りながら進めることにする。その関数を main から呼び出すことにする。
        - Python だと実行するスクリプトを step01.py, step02.py, ... のように変えながら進めるのが簡単だが、Visual Studio 上の C++ で開発していると、main 関数をひとつに固定しておかないと面倒なので。
    - あと、同じ名前のクラスがプロジェクト内に複数存在するとリンクエラーになってしまうので、とりあえず step.hpp というヘッダに押し込めて共有することにした。
        - この先、別のヘッダにクラスを移すときには削除しなければならないかも。コンパイルエラーになるようなケースでは、古いコードは消していくしかない。

### ステップ 3：関数の連結
- NdArrayPrinter クラス
    - 書籍には無いが、NumCpp の NdArray クラスを標準出力するとき、スカラーであっても配列形式で表示されてしまうので、スカラーか否かで出力形式を切り替えるためのヘルパークラスを作成。
- その他
    - 書籍と実行結果をなるべく合わせるため、NdArray 内部で使用するデータ型は当面 double 型とする。
        - ディープラーニングにはそこまでの精度は不要なので、最終的には float にする。
    - 同様の理由で、NdArrayPrinter クラスを作成した。
    - dz 名前空間を using しておくことにした。これに加えて `auto` を使用することで、main 関数内は書籍の Python コードにかなり近づいた。

### ステップ 4：数値微分
- numrical_diff 関数
    - この先 Variant クラスや Function クラスは内部でデータを保持しながら使うことになり const 参照渡しはマッチしないので、通常の参照渡しに変えることになりそう。
    - 取り急ぎ、Function 型の引数についてのみ const を外したが、今後も必要に応じて外していくことにする。
    - …と思ったら、合成関数の微分のところでに、通常の関数 `f` をこの関数に渡すコードが出てきてしまい、そもそも Function 型にしてはダメだと気付く。 `std::function` を使用して、Callable なオブジェクトを受け取れるように変更した。C++ は型に厳密なので、引数で渡す Callableオブジェクトのシグネチャを `Varialbe(Variable)` に限定せざるを得なかったが、この先、Python のダックタイピングをどこまで受け入れられるか不安。
- step04
    - `f()` というシンプルな関数が出てきた。この先も、各ステップの中でこのような関数が出てくると名前が衝突するので、苦肉の策で、各ステップの処理はそれぞれ別の名前空間にすることにした。過去のステップも修正。

### ステップ 5：計算グラフで表す
- このステップでの実装は無し。
- その他
    - フォルダ構成等見直し。stepXX.cpp を作成し、なるべく書籍のサンプルに合わせた構成とした。
    - ソースファイルごとに NumCpp.hpp をビルドしているとこれから先どんどんビルド時間が伸びてしまうので、プリコンパイル済みヘッダを使用することにした。最終的に再度外すかも。
    - DeZeroCpp のメインソース部分は、必要なヘッダファイルを適宜インクルードするようにする。dezero.hpp に DeZeroCpp で用意したヘッダフィあるのインクルード分を並べ、DeZeroCpp を使用する側（step など）からは、dezero.hpp をインクルードさせる。

### ステップ 6：手作業によるバックプロパゲーション
- Variant, Function クラス
    - 初期状態が None のメンバが出てきた。C++ だと、生ポインタなら nullptr 、スマートポインタなら Empty の状態にすべきか。
    - 生ポインタではなくスマートポインタを使うのは大前提だが、最初 unique_ptr にしようとしてエラーになった。コピー代入を行っているので当たり前だった。shared_ptr を使うこととする。
    - スマートポインタを使うと、その変数に初めて値を代入するときにいちいち `std::make_shared` を使わせることになってしまうのが気になる。それが DeZero ライブラリ内であればよいが、step06 のような使用する側のコードであれば意識させたくない。ただ、おそらくこの辺りも最終的にはライブラリに内包される部分と仮定し、サンプルコードになるべく準拠する状態で先へ進む。

### ステップ 7：バックプロパゲーションの自動化
- Function クラス
    - 抽象クラスにしていたが、`Variable::set_creator` でインスタンスを Function 型の shared_ptr に保持しようとすると、抽象クラスのインスタンス化に失敗というエラーが出てしまいうまくいかなかった。ポリモーフィズムのために親クラス型のポインタ／参照でインスタンスを保持するケースで、スマートポインタをどう使うのが正解か不明。
        - とりあえず、抽象クラスをやめることにした。
- Variant, Function クラス
    - Python のようなメモリ構造（参照カウンタでメモリ管理されており、誰も参照しなくなったオブジェクトは自動的に削除される仕組み）をそのまま実現するために、shared_ptr を利用していたが、中途半端にスマートポインタを利用していてはダメで、ライブラリを利用する側が Variable や Function を作ったタイミングでスマートポインタになっていないといけない。これは利用側の負担が大きいため、しばらく生ポインタでアドレスを保持することにして先に進む。
    - 既に存在するオブジェクトを指すポインタは生ポインタとするが、新たに `new` はしたくないので、そこだけは unique_ptr を使う。
    - 言い方を変えると、Function と Variable がお互いを指す場合のポインタは生ポインタを使うことになる。それぞれのオブジェクトのスコープが不一致だと破綻するが、暫定処置とする。
    - 今後、もし shared_ptr を使う場合、`Variable::set_creator` 関数に this を渡す際、this は生ポインタなのでそのまま渡せない。その場合は、`enable_shared_from_this` を利用する必要がある。
    - `Variable::backward` の内部で `Function` クラスのメンバを参照するので、宣言と定義を分離する必要が出てきた。もともとヘッダオンリーのライブラリに使用としていたが、ここで破綻してきた。最終的に h/cpp を分けることとする。
- step07
    - サンプルコードでは assert でインスタンスの同値チェックをしているが、C++ だとアドレスの一致チェックとした。スマートポインタを使う場合は再考が必要かもしれないが、とりあえずは生ポインタとしたので問題なく使える。

### ステップ 8：再帰からループへ
- 結局全部生ポインタにしてしまった。スマートポインタを使うことで、make関数や `*` 演算子を使用しなければならなくなって、数式チックにコードを書けるメリットが損なわれていってしまったので。

