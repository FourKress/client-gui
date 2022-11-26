const state = {
  printInvoiceInfo: '',
};

const getters = {
  printInvoiceInfo(data) {
    return data.printInvoiceInfo;
  },
};

const mutations = {
  setPrintInvoiceInfo(state, value) {
    state.printInvoiceInfo = value;
  },
};

const actions = {
  setPrintInvoiceInfo({ commit }, value) {
    commit('setPrintInvoiceInfo', value);
  },
};

export default {
  namespaced: true,
  state,
  getters,
  mutations,
  actions,
};
