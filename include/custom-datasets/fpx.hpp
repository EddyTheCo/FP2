
#pragma once
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <iostream>
#include<fstream>
namespace custom_models {
	namespace datasets{
		/**
		 * Dataset based on the Featured Proposal FPX
		 * Encoding the adjacency list as features.
		 *
		 */
		class  FPX : public torch::data::Dataset<FPX> {
			public:
				enum class Mode { kTrain, kTest };

			 explicit FPX(const std::string& root, Mode mode = Mode::kTrain,const std::string name_m="");

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

		class FP2 :public FPX
		{
			public:
			 FP2(const std::string& root, Mode mode = Mode::kTrain):FPX(root,mode,"2"){};
		};
		class FP3_1 :public FPX
		{
			public:
			 FP3_1(const std::string& root, Mode mode = Mode::kTrain):FPX(root,mode,"3_1"){};
		};


	};
};
