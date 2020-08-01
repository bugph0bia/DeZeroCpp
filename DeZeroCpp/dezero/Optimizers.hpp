#pragma once

#include "../dezero/dezero.hpp"

namespace L = dz::layers;

namespace dz::optimizers
{

//----------------------------------
// class
//----------------------------------

// 最適化クラス
class Optimizer
{
public:
	// 前処理の関数型
	using hook_t = void(const L::params_t&);

	// 対象レイヤー
	L::LayerPtr target;
	// 前処理コレクション
	std::list<std::function<hook_t>> hooks;

	// 初期設定
	Optimizer& setup(L::LayerPtr target)
	{
		this->target = target;
		return *this;
	}
	Optimizer& setup(L::Layer* target)
	{
		return this->setup(L::LayerPtr(target));
	}

	// 更新処理
	void update()
	{
		// 勾配が設定されているパラメータをまとめる
		L::params_t params;
		for (auto& p : this->target->params()) {
			if (p->grad) params.insert(p);
		}
		// 前処理（オプション）
		for (auto f : this->hooks) {
			f(params);
		}
		// パラメータの更新
		for (auto& param : params) {
			this->update_one(param);
		}
	}

	// パラメータ更新
	virtual void update_one(const VariablePtr& param) = 0;

	// 前処理登録
	void add_hook(std::function<hook_t>& f)
	{
		hooks.push_back(f);
	}
};

// 勾配降下法 (SGD: Stochastic Gradient Descent) クラス
class SGD : public Optimizer
{
public:
	// 学習係数
	double lr;

	// コンストラクタ
	SGD(double lr = 0.01) :
		lr(lr)
	{}

	// パラメータ更新
	void update_one(const VariablePtr& param) override
	{
		// 勾配に学習係数を乗算した値でパラメータを更新
		*(param->data) -= this->lr * *(param->grad->data);
	}
};

// MomentumSGDクラス
class MomentumSGD : public Optimizer
{
public:
	double lr;
	double momentum;
	std::unordered_map<uintptr_t, NdArrayPtr> vs;

	// コンストラクタ
	MomentumSGD(double lr = 0.01, double momentum = 0.9) :
		lr(lr),
		momentum(momentum)
	{}

	// パラメータ更新
	void update_one(const VariablePtr& param) override
	{
		// パラメータのユニークIDを取得
		auto v_key = utils::id(param);
		// IDが未登録ならパラメータと同形状の0テンソルで初期化
		if (this->vs.find(v_key) == this->vs.end()) {
			this->vs[v_key] = as_array(nc::zeros_like<data_t>(*param->data));
		}

		// パラメータを更新
		auto v = this->vs[v_key];
		*v *= this->momentum;
		*v -= this->lr * *param->grad->data;
		*param->data += *v;
	}
};

}	// namespace dz::optimizers
