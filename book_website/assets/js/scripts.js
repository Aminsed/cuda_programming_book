// assets/js/scripts.js
// Include all your custom JavaScript here

// Initialize AOS
AOS.init({
    once: true,
    duration: 800,
});

// Navbar Transition on Scroll
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Smooth Scrolling
document.querySelectorAll('a.nav-link, a.text-decoration-none, .tree a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href && href.startsWith('#')) {
            e.preventDefault();
            document.querySelector(href).scrollIntoView({
                behavior: 'smooth',
                block: 'start',
            });
        }
    });
});

// Back to Top Button
const backToTopBtn = document.getElementById('backToTopBtn');
window.addEventListener('scroll', () => {
    if (window.scrollY > 500) {
        backToTopBtn.style.display = 'block';
    } else {
        backToTopBtn.style.display = 'none';
    }
});
backToTopBtn.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

// Dark Mode Toggle
const darkModeToggle = document.getElementById('darkModeToggle');
darkModeToggle.addEventListener('change', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
});
// Load Theme Preference
window.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
        darkModeToggle.checked = true;
    }
});

// Tree View functionality
document.addEventListener('DOMContentLoaded', function() {
    var toggler = document.getElementsByClassName('caret');
    for (var i = 0; i < toggler.length; i++) {
        toggler[i].addEventListener('click', function() {
            this.parentElement.querySelector('.nested').classList.toggle('active');
            this.classList.toggle('caret-down');
        });
    }

    // Initially collapse all except the first level
    var nestedLists = document.querySelectorAll('.nested');
    nestedLists.forEach(function(list) {
        list.classList.remove('active');
    });
    var carets = document.querySelectorAll('.caret');
    carets.forEach(function(caret) {
        caret.classList.remove('caret-down');
    });
    // Expand the first level
    var firstLevelNested = document.querySelector('.tree > li > .nested');
    if (firstLevelNested) {
        firstLevelNested.classList.add('active');
        var firstCaret = document.querySelector('.tree > li > .caret');
        if (firstCaret) {
            firstCaret.classList.add('caret-down');
        }
    }
});

// Sidebar Toggle Functionality
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const mainContent = document.getElementById('main-content');

sidebarToggle.addEventListener('click', function() {
    sidebar.classList.toggle('collapsed');
    sidebarToggle.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');

    // Change toggle button icon
    const icon = sidebarToggle.querySelector('i');
    if (sidebar.classList.contains('collapsed')) {
        icon.classList.remove('fa-angle-left');
        icon.classList.add('fa-angle-right');
    } else {
        icon.classList.remove('fa-angle-right');
        icon.classList.add('fa-angle-left');
    }
});

// Initialize Bootstrap ScrollSpy
if (mainContent) {
    new bootstrap.ScrollSpy(document.body, {
        target: '#sidebar',
        offset: 100
    });
}