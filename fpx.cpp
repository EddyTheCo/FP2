#include"custom-datasets/fpx.hpp"
namespace custom_models {
	namespace datasets{


		FPX::FPX(const std::string& root, Mode mode,const std::string name): mode_(mode)
		{
			auto pathIM=(root+"/FP_"+name+((mode==Mode::kTrain)?"_train_input.bin":"_test_input.bin"));
			auto pathTA=(root+"/FP_"+name+((mode==Mode::kTrain)?"_train_target.bin":"_test_target.bin"));
			std::ifstream file_image((pathIM).c_str(),std::ios::in | std::ios::binary);
			std::ifstream file_target((pathTA).c_str(),std::ios::in | std::ios::binary);
			TORCH_CHECK(file_image, "Error opening data file at ", pathIM);
			TORCH_CHECK(file_target, "Error opening data file at ", pathTA);
			int32_t NitemsIM,Nfeatures,NitemsTA;
			int16_t Nnodes;
			file_image.read(reinterpret_cast<char *>(&NitemsIM), sizeof(NitemsIM));
			file_image.read(reinterpret_cast<char *>(&Nnodes), sizeof(Nnodes));
			file_target.read(reinterpret_cast<char *>(&NitemsTA), sizeof(NitemsTA));
			file_target.read(reinterpret_cast<char *>(&Nnodes), sizeof(Nnodes));
			file_image.read(reinterpret_cast<char *>(&Nfeatures), sizeof(Nfeatures));
			std::cout<<"n features:"<<(int)Nfeatures<<std::endl;
			std::cout<<"n nodes:"<<(int)Nnodes<<std::endl;
			std::cout<<"Nitems:"<<(int)NitemsTA<<" "<<(int)NitemsIM<<std::endl;
			TORCH_CHECK(NitemsIM==NitemsTA,
					"Number of examples in input and target do not match");

			std::vector<int16_t> data_buffer_IM(NitemsIM*Nfeatures);
			std::for_each(data_buffer_IM.begin(),data_buffer_IM.end(),[&file_image](auto &item){
					file_image.read(reinterpret_cast<char *>(&item), sizeof(item));
					});


			images_ = torch::from_blob(data_buffer_IM.data(),{NitemsIM,Nfeatures},torch::dtype(torch::kInt16)).clone();
			images_=images_.to(torch::kFloat64).div_(Nnodes);

			std::vector<int8_t> data_buffer_TA(NitemsTA);
			std::for_each(data_buffer_TA.begin(),data_buffer_TA.end(),[&file_target](auto &item){
					file_target.read(reinterpret_cast<char *>(&item), sizeof(item));
					});

			targets_ = torch::from_blob(data_buffer_TA.data(),{NitemsTA},torch::dtype(torch::kInt8)).clone();
			file_image.close();
			file_target.close();

		}

		torch::data::Example<> FPX::get(size_t index)
		{
			return {images_[index],targets_[index]};
		}
		c10::optional<size_t> FPX::size() const
		{
			return images_.size(0);
		}

	}
}
