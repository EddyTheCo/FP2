#include"custom-datasets/fp2.hpp"
namespace network {
	namespace datasets{

		FP2::FP2(const std::string& root, Mode mode): mode_(mode)
		{
			auto pathIM=(root+((mode==Mode::kTrain)?"/FP2_train_input.bin":"/FP2_test_input.bin"));
			auto pathTA=(root+((mode==Mode::kTrain)?"/FP2_train_target.bin":"/FP2_test_target.bin"));
			std::ifstream file_image((pathIM).c_str(),std::ios::in | std::ios::binary);
			std::ifstream file_target((pathTA).c_str(),std::ios::in | std::ios::binary);
			TORCH_CHECK(file_image, "Error opening data file at ", pathIM);
			TORCH_CHECK(file_target, "Error opening data file at ", pathTA);
			int32_t NitemsIM,maxEdges,NitemsTA;
			file_image.read(reinterpret_cast<char *>(&NitemsIM), sizeof(NitemsIM));
			file_target.read(reinterpret_cast<char *>(&NitemsTA), sizeof(NitemsTA));
			file_image.read(reinterpret_cast<char *>(&maxEdges), sizeof(maxEdges));
			std::cout<<"maxEdges:"<<(int)maxEdges<<std::endl;
			std::cout<<"Nitems:"<<(int)NitemsTA<<" "<<(int)NitemsIM<<std::endl;
			TORCH_CHECK(NitemsIM==NitemsTA,
					"Number of examples in input and target do not match");

			std::vector<int16_t> data_buffer_IM(NitemsIM*maxEdges);
			std::for_each(data_buffer_IM.begin(),data_buffer_IM.end(),[&file_image](auto &item){
					file_image.read(reinterpret_cast<char *>(&item), sizeof(item));
					});


			images_ = torch::from_blob(data_buffer_IM.data(),{NitemsIM,maxEdges},torch::dtype(torch::kInt16)).clone();
			images_=images_.to(torch::kFloat64).div_(maxEdges);

			std::vector<int8_t> data_buffer_TA(NitemsTA);
			std::for_each(data_buffer_TA.begin(),data_buffer_TA.end(),[&file_target](auto &item){
					file_target.read(reinterpret_cast<char *>(&item), sizeof(item));
					});

			targets_ = torch::from_blob(data_buffer_TA.data(),{NitemsTA},torch::dtype(torch::kInt8)).clone();
file_image.close();
file_target.close();

		}
		torch::data::Example<> FP2::get(size_t index)
		{
			return torch::data::Example(images_[index],targets_[index]);
		}
		c10::optional<size_t> FP2::size() const
		{
			return images_.size(0);
		}

	}
}
