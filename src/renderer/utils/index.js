export const validateInt = (rule, value, callback) => {
  if (value && !/^\d+$/.test(value)) {
    callback('请输入整数');
  }
  callback();
};
export const validateFloat = (rule, value, callback) => {
  if (value && !/^\d+(?:\.\d+)?$/.test(value)) {
    callback('请输入整数或浮点数');
  }
  callback();
};
