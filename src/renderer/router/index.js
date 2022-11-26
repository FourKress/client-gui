import Vue from 'vue';
import Router from 'vue-router';

Vue.use(Router);

const router = new Router({
  routes: [
    {
      name: '首页',
      id: 'home',
      path: '/',
      component: require('../views/home').default,
    },
  ],
});

Vue.use(Router);

export default router;
