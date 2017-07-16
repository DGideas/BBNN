// Bidirectional Bootstrap Neural Network
// 双向自举神经网络
// dgideas@outlook.com
// 2017年07月
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <utility>
#include <cmath>
#include <random>
#include <ctime>

using namespace std;

static auto randomSeed = time(0);

inline int getRandomInteger(const int& _a, const int& _b) noexcept
{
	default_random_engine generator(randomSeed++);
	uniform_int_distribution<int> distribution(_a, _b);
	return distribution(generator);
}

inline double getRandomReal(
	const double& _a=0.0, const double& _b=1.0) noexcept
{
	default_random_engine generator(randomSeed++);
	uniform_real_distribution<double> distribution(_a, _b);
	return distribution(generator);
}

inline double function_signal(const double& _val) noexcept
{
	if (_val > 0)
	{
		return 1;
	}
	else if (_val == 0)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

class NeuralNetworkBase{};

class NeuralNetworkElement
{
	public:
		NeuralNetworkElement();
		~NeuralNetworkElement() = default;
	protected:
		static size_t _idCount;
		size_t _id;
};

size_t NeuralNetworkElement::_idCount = 0;

NeuralNetworkElement::NeuralNetworkElement()
{
	this->_id = NeuralNetworkElement::_idCount++;
	return;
}

template <typename _CalculateType>
class BidirectionalBootstrapNeuralNetwork;
template <typename _CalculateType>
class NeuralUnit;

// 用于神经网络输出层的接收器
template <typename _CalculateType = double>
class OutputReceiver: public NeuralNetworkElement
{
	friend class BidirectionalBootstrapNeuralNetwork<_CalculateType>;
	public:
		OutputReceiver() = delete;
		OutputReceiver(const shared_ptr<NeuralNetworkBase>&) noexcept;
		void makeImpulses(const _CalculateType&,
			const shared_ptr<NeuralUnit<_CalculateType>>&) noexcept;
	private:
		_CalculateType _cache;
};

template <typename _CalculateType>
OutputReceiver<_CalculateType>::OutputReceiver(
	const shared_ptr<NeuralNetworkBase>& _nn) noexcept
{
	//cout<<"对于神经网络"<<_nn<<"，构建了"<<this<<endl;
	return;
}

template <typename _CalculateType>
inline void OutputReceiver<_CalculateType>::makeImpulses(
	const _CalculateType& _impulses,
	const shared_ptr<NeuralUnit<_CalculateType>>& _source) noexcept
{
	return;
}

template <typename _CalculateType = double>
class NeuralUnit:
	public NeuralNetworkElement,
	public enable_shared_from_this<NeuralUnit<_CalculateType>>
{
	friend class BidirectionalBootstrapNeuralNetwork<_CalculateType>;
	public:
		void connectTo(const shared_ptr<NeuralUnit<_CalculateType>>&)
			noexcept;
		void connectedFrom(const shared_ptr<NeuralUnit<_CalculateType>>&)
			noexcept;
		void makeImpulses(const _CalculateType&,
			const shared_ptr<NeuralUnit<_CalculateType>>&) noexcept;
		void punishment() noexcept;
		void printCache() noexcept;
	protected:
		vector<shared_ptr<NeuralUnit<_CalculateType>>> _connectTo;
		vector<shared_ptr<NeuralUnit<_CalculateType>>> _connectedFrom;
		map<shared_ptr<NeuralUnit<_CalculateType>>, _CalculateType>
			_weight;
		_CalculateType _threshold;
		map<shared_ptr<NeuralUnit<_CalculateType>>, _CalculateType>
			_impulsesCache;
};

template <typename _CalculateType>
inline void NeuralUnit<_CalculateType>::connectTo(
	const shared_ptr<NeuralUnit<_CalculateType>>& _unit) noexcept
{
	this->_connectTo.push_back(_unit);
	return;
}

template <typename _CalculateType>
inline void NeuralUnit<_CalculateType>::connectedFrom(
	const shared_ptr<NeuralUnit<_CalculateType>>& _unit) noexcept
{
	this->_connectedFrom.push_back(_unit);
	this->_weight[_unit] = getRandomReal(-1, 1);
	this->_impulsesCache[_unit] = 0;
	return;
}

template <typename _CalculateType>
inline void NeuralUnit<_CalculateType>::
	makeImpulses(const _CalculateType& _impulses,
		const shared_ptr<NeuralUnit<_CalculateType>>& _source) noexcept
{
	// 如果是输入层神经元, 直接向下传递
	if (this->_connectedFrom.empty())
	{
		for (auto& nextLevelNeuralUnit: this->_connectTo)
		{
			auto self =	this->shared_from_this();
			nextLevelNeuralUnit->makeImpulses(_impulses, self);
		}
	}
	else
	{
		this->_impulsesCache[_source] = _impulses;
		_CalculateType judge = (_CalculateType)0;
		for (const auto& weightElement: this->_weight)
		{
			judge += 
				weightElement.second *
				this->_impulsesCache[weightElement.first];
		}
		judge += this->_threshold;
		judge = function_signal(judge);
	}
	return;
}

template <typename _CalculateType>
inline void NeuralUnit<_CalculateType>::printCache() noexcept
{
	cout<<"\t["<<this->_id<<"]的 Cache 状态如下:"<<endl;
	for (const auto& cacheElement: this->_impulsesCache)
	{
		cout<<"\t\tCache["<<cacheElement.first<<"] = "<<
			cacheElement.second<<";"<<endl;
	}
}

template <typename _CalculateType = double>
class BidirectionalBootstrapNeuralNetwork:
	public NeuralNetworkBase,
	public enable_shared_from_this<
		BidirectionalBootstrapNeuralNetwork<
			_CalculateType
		>
	>
{
	public:
		BidirectionalBootstrapNeuralNetwork() = delete;
		BidirectionalBootstrapNeuralNetwork(
			const size_t&, const size_t&) noexcept;
		static shared_ptr<BidirectionalBootstrapNeuralNetwork<_CalculateType>>
			getNetwork(const size_t&, const size_t&) noexcept;
		BidirectionalBootstrapNeuralNetwork(
			const BidirectionalBootstrapNeuralNetwork&) = delete;
		BidirectionalBootstrapNeuralNetwork(
			BidirectionalBootstrapNeuralNetwork&&) = delete;
		BidirectionalBootstrapNeuralNetwork& operator= (
			const BidirectionalBootstrapNeuralNetwork&) = delete;
		vector<shared_ptr<NeuralUnit<_CalculateType>>>& getInputLayer()
			noexcept;
		vector<shared_ptr<NeuralUnit<_CalculateType>>>& getOutputLayer()
			noexcept;
		void training(
			const pair<vector<_CalculateType>,
			vector<_CalculateType>>&) noexcept;
		void print() noexcept;
	protected:

		vector<shared_ptr<NeuralUnit<_CalculateType>>> _inputLayer;
		vector<shared_ptr<NeuralUnit<_CalculateType>>> _outputLayer;
		vector<shared_ptr<NeuralUnit<_CalculateType>>> _neuralUnit;
	private:
		void _setInputLayer(const size_t&) noexcept;
		void _setOutputLayer(const size_t&) noexcept;
		void _initializeTask() noexcept;
		void _initializeConnection() noexcept;
		void _runtimeTask() noexcept;
		bool _taskInitializeConnection;
};

template <typename _CalculateType>
BidirectionalBootstrapNeuralNetwork<_CalculateType>::
	BidirectionalBootstrapNeuralNetwork(
		const size_t& _inputArgumentCount, const size_t& _outputArgumentCount)
	noexcept
{
	this->_setInputLayer(_inputArgumentCount);
	this->_setOutputLayer(_outputArgumentCount);
	this->_initializeTask();
	return;
}

template <typename _CalculateType>
shared_ptr<BidirectionalBootstrapNeuralNetwork<_CalculateType>>
	BidirectionalBootstrapNeuralNetwork<_CalculateType>::
		getNetwork(const size_t& _inputArgumentCount,
			const size_t& _outputArgumentCount) noexcept
{
	auto instance =
		make_shared<BidirectionalBootstrapNeuralNetwork<_CalculateType>>(
			_inputArgumentCount, _outputArgumentCount);
	return instance;
}

template <typename _CalculateType>
inline void
	BidirectionalBootstrapNeuralNetwork<_CalculateType>::_setInputLayer(
		const size_t& _inputArgumentCount) noexcept
{
	for (size_t i=0; i!=_inputArgumentCount; i++)
	{
		auto unit = make_shared<NeuralUnit<_CalculateType>>();
		this->_inputLayer.push_back(unit);
		this->_neuralUnit.push_back(unit);
	}
	return;
}

template <typename _CalculateType>
inline void
	BidirectionalBootstrapNeuralNetwork<_CalculateType>::_setOutputLayer(
		const size_t& _outputArgumentCount) noexcept
{
	for (size_t i=0; i!=_outputArgumentCount; i++)
	{
		auto unit = make_shared<NeuralUnit<_CalculateType>>();
		this->_outputLayer.push_back(unit);
		this->_neuralUnit.push_back(unit);
	}
	return;
}

template <typename _CalculateType>
inline void BidirectionalBootstrapNeuralNetwork<_CalculateType>::
	_initializeTask() noexcept
{
	this->_taskInitializeConnection = true;
	return;
}

// 不能在构造时调用, 因为要获取this的智能指针
// 通过运行时任务进行调用
template <typename _CalculateType>
inline void BidirectionalBootstrapNeuralNetwork<_CalculateType>::
	_initializeConnection() noexcept
{
	for (auto& inputUnitPtr: this->_inputLayer)
	{
		for (auto& outputUnitPtr: this->_outputLayer)
		{
			inputUnitPtr->connectTo(outputUnitPtr);
			outputUnitPtr->connectedFrom(inputUnitPtr);
		}
	}
	// 构建神经网络输出数据接收层
	for (auto& outputUnitPtr: this->_outputLayer)
	{
		auto self = this->shared_from_this();
		//this->_neuralUnit.push_back(unit);
	}
	return;
}

// 运行时初始化检测
template <typename _CalculateType>
inline void BidirectionalBootstrapNeuralNetwork<_CalculateType>::
	_runtimeTask() noexcept
{
	if (this->_taskInitializeConnection)
	{
		this->_initializeConnection();
		this->_taskInitializeConnection = false;
	}
	return;
}

template <typename _CalculateType>
inline vector<shared_ptr<NeuralUnit<_CalculateType>>>&
	BidirectionalBootstrapNeuralNetwork<_CalculateType>::
		getInputLayer() noexcept
{
	this->_runtimeTask();
	return this->_inputLayer;
}

template <typename _CalculateType>
inline vector<shared_ptr<NeuralUnit<_CalculateType>>>&
	BidirectionalBootstrapNeuralNetwork<_CalculateType>::
		getOutputLayer() noexcept
{
	this->_runtimeTask();
	return this->_outputLayer;
}

template <typename _CalculateType>
inline void
	BidirectionalBootstrapNeuralNetwork<_CalculateType>::training(
		const pair<vector<_CalculateType>,
			vector<_CalculateType>>& _trainingItem) noexcept
{
	this->_runtimeTask();
	const auto& _inputArray = _trainingItem.first;
	const auto& _outputArray = _trainingItem.second;
	if (_inputArray.size() != this->_inputLayer.size() ||
		_outputArray.size() != this->_outputLayer.size())
	{
		return;
	}
	for (size_t order=0; order!=_inputLayer.size(); order++)
	{
		auto& inputNeuralUnit = _inputLayer[order];
		auto& impulses = _inputArray[order];
		inputNeuralUnit->makeImpulses(impulses, nullptr);
	}
	return;
}

template <typename _CalculateType>
inline void BidirectionalBootstrapNeuralNetwork<_CalculateType>::
	print() noexcept
{
	this->_runtimeTask();
	for (const auto& unit: this->_neuralUnit)
	{
		cout<<"["<<unit->_id<<"]神经元 "<<unit<<endl;
		if (unit->_connectedFrom.empty())
		{
			cout<<"\t这是输入层神经元，id为"<<unit->_id<<endl;
		}
		else
		{
			cout<<"\t权重为: ";
			for (const auto& weight: unit->_weight)
			{
				cout<<weight.second<<"(来自"<<weight.first<<") ";
			}
			cout<<endl;
			cout<<"\t阈值: "<<unit->_threshold<<endl;
			unit->printCache();
		}
		if (unit->_connectTo.empty())
		{
			cout<<"\t这是输出层神经元，id为"<<unit->_id<<endl;
		}
		else
		{
			cout<<"\t连接到:"<<endl;
			for (const auto& connectToUnit: unit->_connectTo)
			{
				cout<<"\t  *"<<connectToUnit<<endl;
			}
		}
		cout<<"****************"<<endl;
	}
}

int main(int argc, char* argv[])
{
	using BBNN = BidirectionalBootstrapNeuralNetwork<>;
	auto XOR = BidirectionalBootstrapNeuralNetwork<>::getNetwork(2, 1);
	XOR->training({{0, 0}, {0}});
	XOR->training({{0, 1}, {1}});
	XOR->training({{1, 0}, {1}});
	XOR->training({{1, 1}, {0}});
	XOR->print();
	return 0;
}
