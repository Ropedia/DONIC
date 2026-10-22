/* DONIC project page — editable content.
 * Every number on the page comes from here, and every number here comes from
 * the paper (abstract, Table 1, Table 2). Edit, save, reload.
 */
window.DONIC = {

  /* ---------- identity ---------- */
  site: {
    authors: [
      { name: 'Mingze Sun' }, { name: 'Ziang Cao' }, { name: 'Yuan Fei' }, { name: 'Qiankuan Tao' },
      { name: 'Zihan Wang' }, { name: 'Tianqi Liu' }, { name: 'Junhao Chen' }, { name: 'Haiwen Diao' },
      { name: 'Fangzhou Hong' }, { name: 'Zhaoxi Chen' }, { name: 'Ziwei Liu' }
    ],
    affiliations: 'Ropedia',
    anonymousLine: 'Anonymous authors',  // shown only if the author list is emptied
    paperUrl: 'https://arxiv.org/abs/2610.29997', // paper link; empty = button shows "soon"
    codeUrl: 'https://github.com/Ropedia/DONIC',  // code link; empty = Code button hidden
    datasetUrl: '',
    bibtex:
`@article{sun2026donic,
  title   = {DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations},
  author  = {Sun, Mingze and Cao, Ziang and Fei, Yuan and Tao, Qiankuan and Wang, Zihan and Liu, Tianqi and Chen, Junhao and Diao, Haiwen and Hong, Fangzhou and Chen, Zhaoxi and Liu, Ziwei},
  journal = {arXiv preprint arXiv:2610.29997},
  year    = {2026}
}`
  },

  /* ---------- hero readout tape (cycled one cell at a time) ---------- */
  tape: [
    ['checkpoint', '1'],
    ['fine-tuning', '0'],
    ['unseen traj SR', '86.79%'],
    ['unseen obj SR', '57.32%'],
    ['hands', 'L + R']
  ],

  /* ---------- BiHOI, paper §3 and Table 1 ---------- */
  dataset: {
    stats: [
      ['1,575', 'sequences'],
      ['229', 'objects'],
      ['14.77', 'hours', ' h'],
      ['32', 'calibrated RGB cameras']
    ],
    // share of contact frames in which both hands touch (Table 1, last column)
    contact: [
      ['BiHOI (ours)', 65.91],
      ['H2O', 38.57],
      ['GigaHands', 34.74],
      ['GRAB', 32.21],
      ['OakInk2', 24.97],
      ['TACO', 15.69]
    ]
  },

  /* ---------- case viewers (file stems in assets/media) ---------- */
  viewers: {
    capture: {
      columns: [['human', 'Human recording'], ['recon', '3D reconstruction'],
                ['reference', 'Retargeted reference'], ['policy', 'DONIC rollout', true]],
      aspect: '16 / 9',
      cases: [['capture-01', 'Peeler'], ['capture-02', 'Funnel'], ['capture-03', 'Bowl']]
    },
    unseen: {
      columns: [['human', 'Human reference'], ['recon', '3D reconstruction'],
                ['reference', 'Retargeted reference'], ['policy', 'DONIC rollout', true]],
      aspect: '1 / 1',
      cases: [['unseen-01', 'Cup'], ['unseen-02', 'Bottle'], ['unseen-03', 'Brush'],
              ['unseen-04', 'Roller'], ['unseen-05', 'Knife'], ['unseen-06', 'Container']]
    },
    comparison: {
      columns: [['reference', 'Retargeted reference'], ['dextrack', 'DexTrack-bi'],
                ['dexplore', 'Dexplore-bi'], ['policy', 'DONIC', true]],
      aspect: '1 / 1',
      cases: [['comparison-01', 'Brush I'], ['comparison-02', 'Brush II'], ['comparison-03', 'Knife'],
              ['comparison-04', 'Spatula'], ['comparison-05', 'Object inspection']]
    }
  },

  /* ---------- gallery strip: [file stem, label, tag] ---------- */
  gallery: [
    ['unseen-01-policy', 'Cup', 'unseen object'],
    ['capture-01-policy', 'Peeler', 'BiHOI'],
    ['comparison-03-policy', 'Knife', 'comparison'],
    ['unseen-02-policy', 'Bottle', 'unseen object'],
    ['capture-02-policy', 'Funnel', 'BiHOI'],
    ['comparison-04-policy', 'Spatula', 'comparison'],
    ['unseen-03-policy', 'Brush', 'unseen object'],
    ['capture-03-policy', 'Bowl', 'BiHOI'],
    ['comparison-01-policy', 'Brush I', 'comparison'],
    ['unseen-04-policy', 'Roller', 'unseen object'],
    ['comparison-05-policy', 'Object inspection', 'comparison'],
    ['unseen-05-policy', 'Knife', 'unseen object'],
    ['comparison-02-policy', 'Brush II', 'comparison'],
    ['unseen-06-policy', 'Container', 'unseen object']
  ],

  /* ---------- Table 2. Columns: method, To, Ro, Th, Rh, Efinger, SR, Comp ---------- */
  results: {
    trajectory: {
      label: 'Unseen trajectories', clips: 53,
      rows: [
        ['DexTrack-bi',      41.86, 2.3340, 23.03, 0.8296, 0.6301,  0.00,  1.30],
        ['Dexplore-bi',      14.73, 0.8597, 21.88, 0.8412, 0.3166,  9.43, 27.87],
        ['w/o CCA',          11.27, 0.9585,  4.34, 0.2980, 0.0433, 81.13, 62.72],
        ['w/o contact gate', 10.26, 0.9589,  4.00, 0.3013, 0.0338, 81.13, 65.03],
        ['DONIC',            10.27, 0.9277,  3.96, 0.2905, 0.0372, 86.79, 67.62],
        ['DONIC + BiHOI',     9.74, 0.8637,  3.32, 0.2818, 0.0338, 88.68, 68.72]
      ]
    },
    object: {
      label: 'Unseen objects', clips: 164,
      rows: [
        ['DexTrack-bi',     127.87, 2.3532, 72.83, 1.0629, 0.6455,  0.00,  0.88],
        ['Dexplore-bi',      17.92, 1.1431, 19.46, 0.9509, 0.3685, 11.59, 26.37],
        ['w/o CCA',          16.83, 1.1464,  5.52, 0.3235, 0.0378, 48.17, 43.89],
        ['w/o contact gate', 17.61, 1.2041,  4.97, 0.3386, 0.0371, 48.78, 49.52],
        ['DONIC',            15.74, 1.1310,  4.58, 0.3226, 0.0373, 57.32, 54.69],
        ['DONIC + BiHOI',    13.96, 1.1157,  4.54, 0.3206, 0.0361, 59.76, 55.35]
      ]
    }
  }
};
