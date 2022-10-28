
#pragma once
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <iostream>
#include<fstream>
namespace custom_models {
	namespace datasets{
		/**
		 * Dataset based on the Featured Proposal [FP2](../../Proposals/FP_2.md)
		 * Encoding the adjacency list as features.
		 *
		 */

		class  FP2 : public torch::data::Dataset<FP2> {
			public:
				enum class Mode { kTrain, kTest };

				explicit FP2(const std::string& root, Mode mode = Mode::kTrain);

				torch::data::Example<> get(size_t index) override;

				c10::optional<size_t> size() const override;

				bool is_train() const noexcept{
					return mode_ == Mode::kTrain;
				}


				const torch::Tensor& images() const{
					return images_;
				}

				const torch::Tensor& targets() const{
					return targets_;
				}

			private:
				torch::Tensor images_, targets_;
				Mode mode_;
		};

	};
};
