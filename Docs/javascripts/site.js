(function () {
  function initLucide() {
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  }

  function initHomeScrollNav() {
    var nav = document.getElementById('droneHomeNav');
    if (!nav) {
      return;
    }

    var onScroll = function () {
      if (window.scrollY > 42) {
        nav.classList.add('is-scrolled');
      } else {
        nav.classList.remove('is-scrolled');
      }
    };

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  function initSmoothScroll() {
    var home = document.querySelector('.drone-home');
    if (!home) {
      return;
    }

    var anchors = home.querySelectorAll('a[href^="#"]');
    anchors.forEach(function (anchor) {
      anchor.addEventListener('click', function (event) {
        var href = anchor.getAttribute('href');
        if (!href || href.length < 2) {
          return;
        }

        var target = document.querySelector(href);
        if (!target) {
          return;
        }

        event.preventDefault();

        var header = document.querySelector('.md-header');
        var homeNav = document.getElementById('droneHomeNav');
        var offset = 0;

        if (header) {
          offset += header.offsetHeight;
        }
        if (homeNav) {
          offset += homeNav.offsetHeight * 0.45;
        }

        var top = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: Math.max(top, 0), behavior: 'smooth' });
      });
    });
  }

  function boot() {
    initLucide();
    initHomeScrollNav();
    initSmoothScroll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
