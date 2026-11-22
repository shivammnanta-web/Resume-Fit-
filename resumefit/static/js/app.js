/**
 * ResumeFit Interactive Client Application Script
 * Handles tabs, drag-and-drop, presets, evidence span highlighting, search & deletion.
 */

document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initTabs();
  initDropzone();
  initPresets();
  initEvidenceHighlighting();
  initCandidateSearch();
  initDeleteAction();
});

// 1. Tab switching
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      const parent = btn.closest(".tab-container") || btn.closest(".card") || document;

      parent.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      parent.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = parent.querySelector(`#${targetId}`);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });
}

// 2. Drag-and-Drop file dropzones (supports multiple dropzones on page)
function initDropzone() {
  const dropzones = document.querySelectorAll(".dropzone");
  if (!dropzones.length) return;

  dropzones.forEach((dropzone) => {
    const fileInput = dropzone.querySelector("input[type='file']");
    const textHint = dropzone.querySelector(".dropzone-text");

    if (!fileInput) return;

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    dropzone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");

      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        const file = fileInput.files[0];
        if (textHint) {
          textHint.innerHTML = `<strong>Selected file:</strong> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
          textHint.style.color = "var(--teal-glow)";
        }
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) {
        const file = fileInput.files[0];
        if (textHint) {
          textHint.innerHTML = `<strong>Selected file:</strong> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
          textHint.style.color = "var(--teal-glow)";
        }
      }
    });
  });
}

// 3. Demo Presets
const DEMO_PRESETS = {
  python_architect: {
    name: "Alex Mercer",
    jd: `Senior Python & Cloud Architect
Experience: 6+ years required
Degree: Bachelor's or Master's in Computer Science

Responsibilities:
- Architect high-throughput distributed backend services using Python, FastAPI, and PostgreSQL.
- Build event-driven microservices using Docker, Kubernetes, and AWS cloud services.
- Establish robust CI/CD deployment automation and maintain test coverage.
- Collaborate with frontend teams building modern interfaces.

Required Skills: Python, AWS, Docker, Kubernetes, PostgreSQL, Microservices
Preferred Skills: Redis, CI/CD, Terraform, React`,
    resume: `Alex Mercer - Senior Cloud Architect
Email: alex.mercer@example.com | Phone: (555) 019-2834

PROFESSIONAL SUMMARY
Senior Cloud Architect with 7 years of full-lifecycle backend engineering experience.
Specialized in Python microservices, distributed system scalability, AWS cloud deployments, and container orchestration with Docker and Kubernetes.

WORK EXPERIENCE
Senior Backend Architect | CloudScale Technologies (2020 - Present)
- Designed and maintained 15+ production microservices using Python and PostgreSQL handling 20,000 requests per minute.
- Deployed multi-region containerized clusters on AWS EKS using Docker, Kubernetes, and Terraform.
- Built automated CI/CD deployment pipelines reducing release lead times by 60%.
- Integrated distributed cache layers using Redis and message queues.

Software Engineer | Apex Systems (2017 - 2020)
- Developed RESTful APIs in Python and automated data processing jobs.
- Managed relational database schema migrations and query optimizations on PostgreSQL.

EDUCATION
Master of Science in Computer Science | Stanford University (2015 - 2017)
Bachelor of Science in Software Engineering | University of California, Berkeley (2011 - 2015)

TECHNICAL SKILLS
Languages & Frameworks: Python, FastAPI, SQL, Bash
Cloud & Infrastructure: AWS, Docker, Kubernetes, Terraform, Linux
Databases & Caching: PostgreSQL, Redis, MySQL`
  },
  frontend_dev: {
    name: "Samantha Chen",
    jd: `Frontend React Developer
Experience: 2+ years required
Degree: Bachelor's in CS or equivalent

Responsibilities:
- Build responsive, accessible web interfaces in React and TypeScript.
- Implement state management and integrate backend REST APIs.
- Optimize web application performance and client-side caching.

Required Skills: JavaScript, TypeScript, React, HTML, CSS
Preferred Skills: Next.js, Redux, TailwindCSS, Jest`,
    resume: `Samantha Chen - Frontend Developer
Email: samantha.chen@example.com

SUMMARY
Frontend Developer with 3 years of experience crafting interactive single-page web applications using React, TypeScript, and modern CSS.

WORK EXPERIENCE
Frontend Engineer | PixelCraft Studios (2021 - Present)
- Engineered responsive user interfaces for SaaS analytics dashboard using React, TypeScript, and HTML5/CSS3.
- Implemented state management using Redux Toolkit and optimized bundle size by 35%.
- Wrote automated unit and integration tests using Jest and React Testing Library.

Junior Web Developer | InnovateLabs (2020 - 2021)
- Developed client websites using JavaScript, HTML, CSS, and Bootstrap.

EDUCATION
Bachelor of Science in Information Technology | University of Washington (2016 - 2020)

SKILLS
Frontend: JavaScript, TypeScript, React, Next.js, Redux, HTML5, CSS3, TailwindCSS
Tools: Git, Webpack, Figma, Jest`
  }
};

function initPresets() {
  const btnPy = document.getElementById("btn-preset-python");
  const btnFe = document.getElementById("btn-preset-frontend");
  const jdInput = document.getElementById("jd_text");
  const resumeInput = document.getElementById("resume_text");
  const nameInput = document.getElementById("candidate_name");

  if (!jdInput) return;

  function loadPreset(key) {
    const data = DEMO_PRESETS[key];
    if (!data) return;

    jdInput.value = data.jd;
    if (resumeInput) resumeInput.value = data.resume;
    if (nameInput) nameInput.value = data.name;

    // Switch to text tab so user can see pasted text
    const textTabBtn = document.querySelector('.tab-btn[data-tab="tab-text"]');
    if (textTabBtn) textTabBtn.click();
  }

  if (btnPy) btnPy.addEventListener("click", () => loadPreset("python_architect"));
  if (btnFe) btnFe.addEventListener("click", () => loadPreset("frontend_dev"));

  // Compare Page Preset
  const btnCompare = document.getElementById("btn-compare-preset");
  if (btnCompare) {
    btnCompare.addEventListener("click", () => {
      const c1Name = document.getElementById("cand1_name");
      const c1Text = document.getElementById("cand1_text");
      const c2Name = document.getElementById("cand2_name");
      const c2Text = document.getElementById("cand2_text");
      const jd = document.getElementById("jd_text");

      if (jd) jd.value = DEMO_PRESETS.python_architect.jd;
      if (c1Name) c1Name.value = "Alex Mercer";
      if (c1Text) c1Text.value = DEMO_PRESETS.python_architect.resume;
      if (c2Name) c2Name.value = "Jordan Smith";
      if (c2Text) {
        c2Text.value = `Jordan Smith - Junior Developer\nEmail: jordan.smith@example.com\n\nPROFESSIONAL SUMMARY\nJunior Developer with 2 years of experience writing web scripts.\n\nWORK EXPERIENCE\nJunior Web Developer | InnovateLabs (2022 - 2024)\n- Developed small web features in Python and Flask.\n- Created static HTML and CSS pages.\n\nEDUCATION\nBachelor of Arts in English Literature | State College (2018 - 2022)\n\nTECHNICAL SKILLS\nPython, Flask, SQLite, Git, HTML, CSS`;
      }
    });
  }
}

// 4. Interactive Evidence Highlighting
function initEvidenceHighlighting() {
  const chips = document.querySelectorAll(".skill-chip[data-span]");
  const preview = document.querySelector(".resume-preview-pane");

  if (!chips.length || !preview) return;

  // Store original HTML of resume preview
  const originalHtml = preview.innerHTML;

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      // Toggle active chip
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");

      const spanText = chip.getAttribute("data-span");
      let cleanSpan = (spanText && spanText !== "None") ? spanText.trim() : "";
      const skillName = chip.getAttribute("data-skill") || "";

      // If span is empty, fall back to matching the skill name
      if (!cleanSpan && skillName) {
        cleanSpan = skillName.trim();
      }

      if (!cleanSpan) {
        preview.innerHTML = originalHtml;
        return;
      }

      let escaped = cleanSpan.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      let regex = new RegExp(`(${escaped})`, "gi");

      // If full span is not in original HTML, try skill name fallback
      if (!regex.test(preview.textContent) && skillName) {
        escaped = skillName.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        regex = new RegExp(`\\b(${escaped})\\b`, "gi");
      }

      // Replace and scroll
      preview.innerHTML = originalHtml.replace(
        regex,
        '<mark class="evidence-highlight" id="active-evidence-mark">$1</mark>'
      );

      const mark = document.getElementById("active-evidence-mark");
      if (mark) {
        mark.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });
}

// 5. Candidate Pool Search
function initCandidateSearch() {
  const searchInput = document.getElementById("candidate-search");
  const table = document.querySelector(".data-table");

  if (!searchInput || !table) return;

  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    const rows = table.querySelectorAll("tbody tr");

    rows.forEach((row) => {
      const text = row.innerText.toLowerCase();
      if (text.includes(query)) {
        row.style.display = "";
      } else {
        row.style.display = "none";
      }
    });
  });
}

// 6. Right-to-Erasure AJAX Deletion
function initDeleteAction() {
  document.querySelectorAll(".btn-delete-candidate").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      const candId = btn.getAttribute("data-id");
      const candName = btn.getAttribute("data-name") || "this candidate";

      if (!confirm(`Are you sure you want to permanently erase all data and audit vectors for ${candName}? This action cannot be undone.`)) {
        return;
      }

      try {
        const resp = await fetch(`/api/resumes/${candId}`, {
          method: "DELETE",
        });

        if (resp.ok) {
          const row = btn.closest("tr");
          if (row) {
            row.style.opacity = "0";
            setTimeout(() => row.remove(), 300);
            alert(`Right-to-Erasure complete: Candidate ${candName} was permanently erased.`);
          } else {
            alert(`Right-to-Erasure complete: Candidate ${candName} was permanently erased. Redirecting to dashboard.`);
            window.location.href = "/dashboard";
          }
        } else {
          const err = await resp.json();
          alert(`Failed to erase candidate: ${err.error?.message || "Unknown error"}`);
        }
      } catch (err) {
        alert(`Error communicating with server: ${err.message}`);
      }
    });
  });
}

// 7. Dark/Light Mode Theme Toggle
function initThemeToggle() {
  const toggleBtn = document.getElementById("theme-toggle-btn");
  if (!toggleBtn) return;

  const currentTheme = () => document.documentElement.getAttribute("data-theme") || "dark";
  const iconSpan = document.getElementById("theme-toggle-icon");

  const updateIcon = (theme) => {
    if (iconSpan) {
      iconSpan.textContent = theme === "light" ? "☀️" : "🌙";
    }
  };

  updateIcon(currentTheme());

  toggleBtn.addEventListener("click", () => {
    const nextTheme = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("resumefit-theme", nextTheme);
    updateIcon(nextTheme);
  });
}

