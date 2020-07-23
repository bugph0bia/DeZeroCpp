# DeZeroCpp
ゼロから作るDeep Learning ❸ をC++で実装する。

## 概要
「[ゼロから作るDeep Learning ❸](https://www.oreilly.co.jp/books/9784873119069/)」を参考にしながら、ディープラーニング用フレームワーク DeZero を C++ で実装しながら学習する。  

## はじめに
目的は、「ゼロから作るDeep Learning ❸」を再読しながら理解を深めることと、C++ の復習。  

### 環境構成
DeZero で使用している Python 用のライブラリや外部ツールの全てを C++ で用意することは難しいかもしれないが、取り合えず以下の構成で始めてみる。  
Windows 上で Visual Studio 2017 以降を使用する。  
言語は C++17 以降。

| DeZero     | DeZeroCpp                                                                                       |
|------------|-------------------------------------------------------------------------------------------------|
| Python 3   | C++ (MSVC)                                                                                      |
| NumPy      | [NumCpp](https://github.com/dpilger26/NumCpp) (v2.00) + [boost](https://www.boost.org/) (v1.73) |
| matplotlib | [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) ※対応保留中                           |
| CuPy       | 未定。多分 GUI 対応は不可能。                                                                   |
| Pillow     | 未定                                                                                            |
| Graphviz   | Graphviz                                                                                        |

- NumCpp には boost も必要。ただし、libファイルは不要なのでダウンロードしてらビルドせずに配置するだけで良い。
    - プロジェクトのプロパティで、「追加のインクルードディレクトリ」に 2 つのライブラリのパスを追加しておく。
- Graphviz はテキストインタフェースのツールなのでそのまま使える。

### 方針
- NumCpp や boost のように、ヘッダオンリーのライブラリを目指す。
    - step07 で破綻した。最終的に、h/cpp にクラスの宣言と定義を分けることにする。
    - step27 で再挑戦して成功。
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

### 中間検討
#### 試行錯誤
- 生ポインタを使うと何が問題か
    - Function や Variantのインスタンス内部で生ポインタでお互いを参照する状態になる。インスタンスのコピーが行われると、メモリ管理が不可能で誰が delete すべきか決められない。
        - ×：生ポインタを使用してはダメ。
- スマートポインタを使うと何が問題か
    - ライブラリ使用する側のコードも含め、全ての Function や Variant のインスタンスをスマートポインタで管理する必要がある。FunctionやVariantのインスタンスを作るとき、いちいちヘルパー関数 std::make_xxx を使わないといけなくなって使用感が損なわれる。
        - △：ステップが進むと、Function のラッパー関数を作ることになるなど、このあたりはライブラリ内に隠蔽されるから大丈夫かもしれない。Variable を作るときはヘルパー関数を使うしかないが。
    - Function や Variant のインスタンスを利用して計算式を連結するのに自然なコードでなくなる。いちいち * や & 演算子で参照しないといけないなど。これが一番の課題。
        - ○：計算式は NdArray に対してしか作らないのでは？もしそうなら、NdArray は常にコピー渡しとし、Function と Variable はスマートポインタを使用すれば対応可能かもしれない。Function 呼ぶときだけ * や & が必要になってしまう問題もあるが、前述の通り Function のラッパー関数によって隠蔽できる。
    - Function は this を Variant に渡すことになるが、通常、スマートポインタ内部で this は使えない。
        - ○：`std::enable_shared_from_this` が使える。
- ラッパークラスを使い、スマートポインタを内部に隠蔽する方法もあるか
    - Function の派生クラスを自然に作れなくなるなど、ポリモーフィズムがうまくいかなくなる。
        - ×：インナークラスを継承させれば解決できそうだが、Functionを継承して新しいクラスが簡単に作れるようにしたいので複雑にしたくない。

#### 結論
- NdArray は基本的にコピー渡し。`Variable::grad` のように空の状態が必要な場合は `std::shared_ptr` で管理。
- Function と Variable は `std::shared_ptr` で管理。
    - 通常のコンストラクタは `protected` で外から使えなくして、それぞれインスタンス生成用の静的メンバ関数を用意しておき、その中で `std::shared_ptr` に入れて返すようにすることも、CRTP を使えばできそうだが、そこまではやらない。
- Function は `std::enable_shared_from_this` を継承。
- Function の 派生クラスを `std::shared_ptr<Function>` に入れる場合、`std::make_shared<Function>` ヘルパー関数は使えないので、new でインスタンス生成する。
    - これを守れば、Function クラスがインスタンス化されることはなくなるので、以前のように抽象クラスに戻す。

### ステップ 9：関数をより便利に
- square, exp 関数
    - ラッパー関数ができたことで、`std::shared_ptr` を使うことによるコードの煩雑さを隠蔽できた。
- as_array, as_variant 関数
    - スカラーを NdArray に変換するためのヘルパー関数であるが、`make::shared_ptr` を内部に隠蔽するためにも使うことにしたので、オーバーロードで様々なバリエーションを用意した。
    - as_variant は書籍にはないが、同様の理由で作成した。
- その他
    - 書籍の後半に出てくる「ndarray だけを扱う」に関しては、C++はそもそも型に厳密なので不要であった。
    - NdArray は基本的にコピー渡しのつもりだったが、結局 Empty (nullptr) の状態を作れなければならず、スマートポインタに統一することになった。

### ステップ 10：テストを行う
- このステップは省略する。

### ステップ 11：可変長の引数（順伝播編）
- Variable::backward 関数
    - 書籍のサンプルコードは、この時点ではリスト化に対応していない（ステップ 13 で対応。）Python だと呼び出ささなければ問題無いが、C++ ではコンパイルが通らないので一時的にコメントアウトした。
- Function::operator() 関数
    - １要素の場合とリストの場合で関数をオーバーロードして対応。
- その他
    - リストは、ランダムアクセスが必要なので `std::vector` を使用する。一部、ソートが必要になる場合は `std::list` で対応する。

### ステップ 12：可変長の引数（改善偏）
- Function::operator() 関数
    - 前のステップで、`std::list` を引数に取る関数を準備したが、これだけで `{}` による `std::initializer_list` の引数渡しに対応できているため、そのままで良い。
    - 戻り値を、１要素とリストの両方に対応することは C++ ではできないので、リストを返すままとする。
- Function::forward 関数
    - リストのアンパッキングに相当する機能は C++ には無いため変更しない。
- 結局、行ったのは add 関数の実装だけ。

### ステップ 13：可変長の引数（逆伝播偏）
- add, square 関数
    - Add, Square クラスの operator() 関数はリストを受け取りリストを返せるように実装した。実際に使う引数や戻り値がリストではなく１要素である場合、これらラッパー関数のシグネチャで対応する。

### ステップ 14：同じ変数を繰り返し使う
- Variable::backward 関数
    - 付録Aに書かれているように、`x->grad` のインスタンスを新しく生成する必要がある。スマートポインタを使っているため、書籍に書かれているように `+=` ではなく `+` を使うだけでは解決にならないため、`as_array` を使ってインスタンス生成するようにした。
- Variable::cleargrad, as_array 関数
    - スマートポインタで管理している変数 grad を空にするために、引数なしの `as_array` 関数を用意していたが、これは不要で、単に `nullptr` を代入すればよかった。こちらの方が書籍のサンプルコードにも近いし簡潔に書けるので、修正した。

### ステップ 15：複雑な計算グラフ（理論編）
- このステップでの実装は無し。

### ステップ 16：複雑な計算グラフ（実装編）
- Function::operator() 関数
    - コンテナから最大値を探すために、`std::max_element` を使用。
- Variable::backward 関数
    - 内部関数（クロージャ）は、ラムダ式で作成。
- Variable::set_creator 関数
    - 内部で Function クラスのメンバを参照するようになったので、宣言と定義を分離した。

### ステップ 17：メモリ管理と循環参照
- step17
    - 弱参照を使用する前は循環参照によってメモリが増え続ける状態となり、弱参照を使用した後はループしてもメモリが増えることが無くなった。
        - プロセスが使用するワーキングセットプライベートの容量で確認。
- その他
    - Python の `weakref` の代わりに `std::weak_ptr` を使用する。

### ステップ 18：メモリ使用量を減らすモード
- Config クラス
    - サンプルコードに倣い文字列でパラメータ名を指定させるために、パラメータの実体を `std::map<std::string, bool>` で保持。
        - bool 以外のパラメータが出てきたらテンプレート化するなど検討する。
    - パラメータの初期化タイミングを保証したかったので、静的クラスではなくシングルトンとした。
        - シングルトンならではのコードはライブラリ内に隠蔽されることを期待。このステップでは問題なし。
- UsingConfig, no_grad クラス
    - いずれも、サンプルコードでは with 句で使用できる関数だったが、C++ ではクラスで作成した。
    - C++ には with 句は無いため、専用のスコープを作って先頭でこれらのクラスをインスタンス化する。そうすることで、前処理はコンストラクタ、後処理はデストラクタに行わせることができ、擬似的に with 句と同じ効果を得ることができる。
    - no_grad の方はあえて class ではなく struct とし、サンプルコードと同様のスネークケースで命名した。
        - 単なるこだわりでしかない。

### ステップ 19：変数を使いやすく
- Variable クラス
    - shape と size は実装。内部の NdArray に委譲すればよく最小限のコードとする。戻り値も型推論で書く。
    - ndim は NumCpp だと 2 固定なので実装不要。
    - dtype は NumCpp だと `NdArray<T>::value_type` で取得可能なので実装不要。
    - len に相当するメンバが NumCpp に無いので未実装。
    - print に相当する << 演算子に対応。

### ステップ 20：演算子のオーバーロード (1)
- VariablePtr クラス
    - add, mul 関数は `VariablePtr` を引数に取るように作っているため、`VariablePtr` の演算子オーバーロードを行うことにした。
        - つまり、`std::shared_ptr<Variable>` の演算子オーバーロードをすることになるが、スマートポインタの算術演算子はオーバーロードされていないため問題無いと考える。
    - 演算子オーバーロードは、定石に従って += 演算子を定義して + 演算子で利用しようとしたが、スマートポインタ、つまり自作していないクラスの演算子オーバーロードをするため、メンバ関数として定義する必要がある += 演算子は使えない。よって、単独で + 演算子をオーバーロードすることとした。* も同じ。
        - そもそも、add, mul 関数を利用して定義する以上、+= や *= はサンプルコードでも使わない前提と思われる。

### ステップ 21：演算子のオーバーロード (2)
- as_variant 関数
    - 以前のステップで既に必要が生じて作成済みなので、ここでは何もする必要が無い。

- VariablePtr クラス
    - 整数, 浮動小数, NdArrayPtr との演算は、C++ だと演算子オーバーロードを複数種類定義することで対応することになる。
        - 内部データ型 `data_t` つまり浮動小数用の演算子オーバーロードを行うことで、暗黙変換によって整数値も使えるようになる。
        - NdArrayPtr 用は個別で定義。

### ステップ 22：演算子のオーバーロード (3)
- VariablePtr クラス
    - 負数(-)演算子は、本来はメンバ関数としてオーバーロードすべきだが、スマートポインタを使う関係上不可能なので、グローバル関数のスタイルで定義。
        - 書籍には記載が無いが、合わせて正数(+)演算子も定義。値に何も変化をもたらさないので、順伝播は入力値をそのまま返し、逆伝播は 1 を返すものとする。実際に処理する関数名は Pos (Positive) とする。
- その他
    - Function クラスの派生クラスである、Add, Sub, Mul, Div などの順伝播／逆伝播の計算は、スマートポインタから中身の `NdArray` まで取り出してしまった方が計算式が簡潔になることに気付いたので修正した。
    - C++ には ** 演算子は無いため、累乗を行う場合は pow 関数を直接使うこととする。

### ステップ 23：パッケージとしてまとめる
- 現時点のライブラリ部分のソースは、core_simple.hpp, core_simple.cpp としてまとめることにした。
    - できればヘッダーオンリーにしたいが、テンプレートではない関数定義がある以上無理みたいなので、ソースファイルも作成。
    - 各ステップのソースに個別の名前空間を切っていたおかげで、同名のクラスや関数を dz 名前空間に定義しても競合することはなかった。
- ライブラリ使用側からは、dezero.hpp をインクルードするだけ（加えて、`using namespace dz` してもよい）で使用できるようになった。（当初の予定通り）
    - dezero.hpp で、core.hpp と core_simple.hpp の切り替えを定義した。

### ステップ 24：複雑な関数の微分
- 計算結果は書籍と一致したので問題無いはず。
- やはり累乗が pow なのが惜しい。^ を使うことも考えたが、演算子の優先順が異なるので問題あり。

### ステップ 25：計算グラフの可視化 (1)
- このステップでの実装は無し。

### ステップ 26：計算グラフの可視化 (2)
- dot_var, dot_func 関数
    - 文字列の書式化に `std::format` を使いたかったが、C++20 以降しか使えないので断念。
- plot_dot_graph 関数
    - dot ファイルの内容も確認したいので、画像ファイルと同じ場所に出力することとした。
    - `std::filesystem::path` を使いたかったので、コンパイル設定を C++17 に設定。
- その他
    - 出力した計算グラフは、書籍とはノードの位置が若干異なるが（Graphviz のバージョンの差異と思われる）、内容は一致。

### ステップ 27：テイラー展開の微分
- power 関数
    - pow とすると、cmath の pow と名前が重なるため変更する。NumCpp にも power が定義されているが、シグネチャが異なるため pow よりは紛らわしくないと判断した。
- my_sin 関数
    - 階乗計算はオーバーフローしてしまうので、前の項の値を利用して計算するようにした。
        - その際、VariablePtr 変数とそれ以外の定数の計算を別に行っておき後でまとめるように注意。
            - 不要な計算グラフを作らないため。
        - Python3 では数値の上限が無いため問題無い模様。
    - ↑を行うと勾配の計算が合わなかったため再検討。計算グラフがサンプルと異なるため？
        - ある程度までオーバーフローしないように double 型を使い、結局は書籍のサンプルと同じコードとした。

### ステップ 28：関数の最適化
- 特になし。

### 中間検討
- `inline` と `static` を適切に指定することで、ヘッダーオンリーにできた。
- `std::tuple` や `std::apply` や、C++17 で導入された機能を使うことでタプルと変数の行き来を自然に行うことができ、よりサンプルコードに近づくことができるかもしれないが、実装までは行っていない。

### ステップ 29：ニュートン法を用いた最適化（手計算）
- 特になし。

### ステップ 30：高階微分（準備編）
- このステップでの実装は無し。

### ステップ 31：高階微分（理論編）
- このステップでの実装は無し。

### ステップ 32：高階微分（実装偏）
- その他
    - core.hpp と core_simple.hpp の切り替えは C++ では難しそう。
        - `Variable::grad` の型が変わることで、各種関数の引数や戻り値の型が変わるため実装し直しになり、それらの関数シグネチャが変わるため過去のステップのコードがコンパイルエラーになるという流れ。型厳密である以上、これらのヘッダを同一視することは難しい。
        - コンパイルエラーになるステップのコード全てに対して、２種類のコードを用意した。

### ステップ 33：ニュートン法を使った最適化（自動計算）
- Variable::backward 関数
    - 設定済みの勾配に新しい勾配を加算するところを、NdArray 同士の計算にしてしまっていたため高階微分が計算されていなかった。VariablePtr 同士の計算にして計算グラフを構築することが重要であった。

### ステップ 34：sin 関数の高階微分
- その他
    - ライブラリのヘッダファイルのインクルード順が重要になるので、全て dezero.hpp に集約して、どのファイルも dezero.hpp をインクルードするようにした。
    - matplotlib でのグラフ描画は、matplotlib-cpp を使わせてもらうかと思ったが、うまく動作させることができず一旦保留。

### ステップ 35：高階微分の計算グラフ
- その他
    - 8階微分で graphviz が途中で終了してしまい、png ファイルを作成できなかった。dot ファイル出力は問題無さそうに見える。最新バージョンを入れて試してみると変わるかもしれない。

### ステップ 36：高階微分以外の用途
- 特になし。

### ステップ 37：テンソルを使う
- NdArrayPrinter の << 演算子
    - NdArray がテンソルの場合（1x1 のスカラでは無い場合）にバグがあったので修正。これまではスカラしか動かしていなかったため気付いていなかった。
- ステップ
    - sum 関数が未実装のため、保留にして先に進む。

### ステップ 38：形状を変える関数
- Variable::reshape 関数
    - この関数の実装のために、Variable クラスの this を使用する必要が出てきたため、Function クラスと同様、`std::enable_shared_from_this` を継承する必要が出てきた。
    - サンプルコードとは違い、C++ 引数がスカラ、タプル、リストのようなケースに対応する必要は無く `nc::Shape` のみで問題無い。
- Variable::transpose 関数
    - C++ ではプロパティ T への対応は行えない。
- その他
    - `NdArray::shape` 関数を変数と誤解してしまうことが多いので注意する。

