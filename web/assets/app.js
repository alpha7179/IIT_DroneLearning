(function () {
  var body = document.body;
  var header = document.getElementById('siteHeader');

  function initIcons() {
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  }

  function initActiveNav() {
    var page = body.dataset.page;
    if (!page) return;

    var links = document.querySelectorAll('[data-route]');
    links.forEach(function (link) {
      if (link.dataset.route === page) {
        link.classList.add('is-active');
      }
    });
  }

  function initHeaderScroll() {
    if (!header) return;

    var onScroll = function () {
      if (window.scrollY > 40) {
        header.classList.add('is-scrolled');
      } else {
        header.classList.remove('is-scrolled');
      }
    };

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  function initSmoothAnchors() {
    var links = document.querySelectorAll('a[href^="#"]');
    links.forEach(function (link) {
      link.addEventListener('click', function (event) {
        var href = link.getAttribute('href');
        if (!href || href.length < 2) return;

        var target = document.querySelector(href);
        if (!target) return;

        event.preventDefault();
        var offset = header ? header.offsetHeight - 8 : 0;
        var top = target.getBoundingClientRect().top + window.scrollY - offset;

        window.scrollTo({ top: Math.max(top, 0), behavior: 'smooth' });
      });
    });
  }

  function initReveal() {
    var revealEls = document.querySelectorAll('.reveal');
    if (!revealEls.length) return;

    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
    );

    revealEls.forEach(function (el) {
      io.observe(el);
    });
  }

  function initTilt() {
    if (window.matchMedia('(max-width: 900px)').matches) {
      return;
    }

    var cards = document.querySelectorAll('.tilt-card');
    cards.forEach(function (card) {
      card.addEventListener('mousemove', function (event) {
        var rect = card.getBoundingClientRect();
        var px = (event.clientX - rect.left) / rect.width;
        var py = (event.clientY - rect.top) / rect.height;

        var rotateY = (px - 0.5) * 8;
        var rotateX = (0.5 - py) * 8;

        card.style.transform = 'perspective(900px) rotateX(' + rotateX.toFixed(2) + 'deg) rotateY(' + rotateY.toFixed(2) + 'deg)';
        card.style.boxShadow = '0 32px 66px -38px rgba(15, 23, 42, 0.5)';
      });

      card.addEventListener('mouseleave', function () {
        card.style.transform = '';
        card.style.boxShadow = '';
      });
    });
  }

  function initHeroParallax() {
    var viewer = document.querySelector('.hero-3d-wrap spline-viewer');
    if (!viewer) return;

    var ticking = false;
    var onScroll = function () {
      if (ticking) return;
      ticking = true;

      window.requestAnimationFrame(function () {
        var y = Math.min(window.scrollY * 0.08, 26);
        viewer.style.transform = 'scale(1.45) translateY(' + y.toFixed(2) + 'px)';
        if (window.matchMedia('(max-width: 900px)').matches) {
          viewer.style.transform = 'scale(1.25) translateY(' + (y * 0.5).toFixed(2) + 'px)';
        }
        ticking = false;
      });
    };

    window.addEventListener('scroll', onScroll, { passive: true });
  }

  function initMermaid() {
    if (!window.mermaid) return;

    window.mermaid.initialize({
      startOnLoad: true,
      securityLevel: 'loose',
      theme: 'base',
      themeVariables: {
        primaryColor: '#e7efff',
        primaryTextColor: '#0f172a',
        primaryBorderColor: '#3b5ea3',
        lineColor: '#1b2f59',
        secondaryColor: '#f8fbff',
        tertiaryColor: '#f1f5f9'
      }
    });
  }

  function boot() {
    initIcons();
    initActiveNav();
    initHeaderScroll();
    initSmoothAnchors();
    initReveal();
    initTilt();
    initHeroParallax();
    initMermaid();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
