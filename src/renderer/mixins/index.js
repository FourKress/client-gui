import { ipcRenderer } from 'electron';

export default {
  methods: {
    onSelect(key, type = 'openFile') {
      ipcRenderer.send('open-directory-dialog', type);
      ipcRenderer.once('selectFilePath', (e, file) => {
        if (!file) return;
        const list = this.form[key];
        if (list.some((d) => d === file)) {
          this.$message.warning('已有相同路径');
          return;
        }
        this.form[key].push(file);
        this.$refs.form?.validateField(key);
      });
    },

    onSelectOnly(key, type = 'openFile') {
      ipcRenderer.send('open-directory-dialog', type);
      ipcRenderer.once('selectFilePath', (e, file) => {
        if (!file) return;
        this.form[key] = file;
        this.$refs.form?.validateField(key);
      });
    },

    handlePathClose(path, key) {
      this.form[key] = this.form[key].filter((d) => d !== path);
      this.$refs.form?.validateField(key);
    },

    handlePathCloseOnly(key) {
      this.form[key] = '';
      this.$refs.form?.validateField(key);
    },
  },
};
