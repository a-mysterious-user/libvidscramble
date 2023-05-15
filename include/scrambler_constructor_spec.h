#pragma once

#include "scrambler.h"
#include <nlohmann/json.hpp>
#include <tuple>


template<typename ValT>
struct pp_kv{
    explicit pp_kv(const std::string &key) : key{key} {}

    const std::string key;
    using val_t = ValT;
};

// constructor specs
template<typename ScramblerT, typename... ValT>
struct pp_cs  {
    using scrambler_t = ScramblerT;
    using kv_t = std::tuple<pp_kv<ValT>...>;
    std::tuple<pp_kv<ValT>...> kvs;
};

template<typename ScramblerT, typename... ValT>
pp_cs<ScramblerT, ValT...> build_scrambler_constructor_spec(const pp_kv<ValT>&... kvs) {
    return pp_cs<ScramblerT, ValT...>{std::make_tuple(kvs...)};
}

template<typename ScramblerT, typename... ValT>
std::shared_ptr<ScramblerBase> construct_scrambler(const nlohmann::json &step_obj, const pp_cs<ScramblerT, ValT...> &spec){
    constexpr const size_t num_param = std::tuple_size<typename pp_cs<ScramblerT, ValT...>::kv_t>::value;
    return construct_scrambler_helper<ScramblerT,pp_cs<ScramblerT, ValT...>>(
            step_obj,
            spec,
            std::make_index_sequence<num_param>{}
            );
}

template<typename ScramblerT,
        typename CSpecT,
        size_t... I>
std::shared_ptr<ScramblerBase> construct_scrambler_helper(const nlohmann::json &step_obj, const CSpecT &spec, std::index_sequence<I...>){
    //const auto &key = std::get<Indices>(spec.kvs).key;
    try{
        auto ptr = std::make_shared<ScramblerT>(
                step_obj[std::get<I>(spec.kvs).key].get<typename std::tuple_element<I, typename CSpecT::kv_t>::type::val_t>()...
            );
        return std::static_pointer_cast<ScramblerBase>(ptr);
    } catch (const std::exception &e) {
        throw std::runtime_error{format("failed to construct scrambler instance from JSON;"
                                 "check if all required arguments in the constructor are provided, "
                                 "and also check if the type of each argument is correct.\n"
                                 "detailed exception information:\n{}", e.what())};
    }
}


// defines the constructor information for each scrambler

const auto RowShuffle_cspec = build_scrambler_constructor_spec<RowShuffle>(
        pp_kv<int>("row_group_size"),
        pp_kv<int>("random_seed"));

const auto ImageTranspose_cspec = build_scrambler_constructor_spec<ImageTranspose>();

const auto RowMix_cspec = build_scrambler_constructor_spec<RowMix>(
        pp_kv<int>("row_group_size"),
        pp_kv<int>("random_seed"));

const auto ImageShift_cspec = build_scrambler_constructor_spec<ImageShift>(
        pp_kv<int>("sx"),
        pp_kv<int>("sy"));