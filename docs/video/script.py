"""
The tutorial, beat by beat. Each beat is one narrated sentence (or two short
ones) with the screen state it plays over and what the viewer's eye is pointed
at. Rect names refer to shots.json produced by capture.py.

  shot   : name of a captured state, a card, or ('seq', prefix, n, fps)
  say    : narration (also the caption)
  hl     : rect names to outline
  click  : rect name the cursor moves to and clicks
  point  : rect name the cursor moves to without clicking
  zoom   : rect name to zoom into (None = full frame)
  label  : short callout next to the first highlight
  hold   : extra seconds after the narration
"""

BEATS = [
    # ------------------------------------------------------------ opening
    dict(shot='card_title', say="A quick tour of the E-Textile Validation GUI.", hold=0.6),
    dict(shot='card_pipeline',
         say="This tool checks how well a motion-sensing garment works, without a motion-capture lab and without writing any code."),
    dict(shot='card_pipeline',
         say="You film a person wearing the garment with an ordinary webcam, while the garment's sensors are logged at the same time."),
    dict(shot='card_pipeline',
         say="The tool estimates the body pose from the video, turns it into joint angles, and trains a model that predicts those angles from the sensors alone."),
    dict(shot='card_files',
         say="Each recording, which we call a session, needs three files: the video, the sensor CSV, and a CSV with the time of every video frame."),
    dict(shot='card_files',
         say="Both CSVs are stamped with the same clock, so the two streams can be lined up exactly."),

    # ------------------------------------------------------------ Tab 1
    dict(shot='tab1_empty', hl=['tabbar'],
         say="Everything happens in five tabs, used from left to right. We start in Data Import."),
    dict(shot='tab1_empty', click='add_train', hl=['add_train'], zoom='add_train', label='Add Subject',
         say="Click Add Subject under Training Subjects."),
    dict(shot='dialog_empty', hl=['browse'], zoom='dialog',
         say="Choose the three files of one session with the Browse buttons."),
    dict(shot='dialog_filled', hl=['report'], zoom='dialog',
         say="The dialog checks the files straight away."),
    dict(shot='dialog_swapped', hl=['report'], zoom='dialog', label='Explained before you continue',
         say="If something is wrong, for example the two CSVs chosen the wrong way round, it says so and explains how to fix it."),
    dict(shot='tab1_tables', hl=['train_table', 'test_table'],
         say="Add as many sessions as you need. Test sessions are held back from training, and only used to score the model."),
    dict(shot='tab1_tables', click='remove_train', hl=['remove_train'], zoom='remove_train',
         say="Remove Selected takes a session out again."),
    dict(shot='tab1_tables', click='extract', hl=['extract'], label='Extract Skeleton & Compute Angles',
         say="When the sessions are in, press Extract Skeleton and Compute Angles."),
    dict(shot='extract_progress', hl=['dialog'], zoom='dialog',
         say="A progress window shows how far it has got and how long is left. You can cancel at any time."),
    dict(shot='tab1_done', hl=['log'], zoom='log',
         say="Results are saved next to each video, so a recording is processed only once. The log reports the frame rate, the sensor channels, and how much time both streams cover."),

    # ------------------------------------------------------------ Tab 2
    dict(shot=('seq', 'tab2_f', 35, 12),
         say="The Skeleton Viewer lets you check the tracking before training anything. The video and the reconstructed skeleton move together."),
    dict(shot='tab2_f17', point='play', hl=['play', 'slider'], zoom='slider',
         say="Press Play, or drag the slider, to move through the recording."),
    dict(shot='tab2_still', hl=['skeleton'], zoom='skeleton',
         say="Each joint is coloured by how confident the tracker is: green is reliable, red means that part of the body was probably hidden."),
    dict(shot='tab2_still', click='tq', hl=['tq'], zoom='tq',
         say="Tracking Quality summarises the whole recording."),
    dict(shot='tab2_tracking', hl=['dialog'], zoom='dialog',
         say="It lists the landmarks and the joint angles that were least reliable, so you know which results to treat with caution."),

    # ------------------------------------------------------------ Tab 3
    dict(shot='tab3_none', hl=['angles'],
         say="In Angles and Sensors you decide what the model will learn."),
    dict(shot='tab3_selected', click='target_box', hl=['angles'], zoom='angles', label='Targets',
         say="Tick the joint angles to predict. The number after each name is its tracking confidence, and low values are shown in red."),
    dict(shot='tab3_selected', hl=['quick'], zoom='quick',
         say="All, None and Shoulder are quick selections."),
    dict(shot='tab3_selected', hl=['sensors'], zoom='sensors', label='Inputs',
         say="Below, choose which sensor channels the model may use as input."),
    dict(shot='tab3_selected', hl=['plots'],
         say="The plots show the selected angles above and the sensors below, on one shared time axis, so you can see by eye whether a sensor responds to the movement."),
    dict(shot='tab3_selected', hl=['info'], zoom='info',
         say="The line underneath gives the time window both recordings cover. Only that part is used for training."),

    # ------------------------------------------------------------ Tab 4
    dict(shot='tab4_ready', hl=['summary'], zoom='summary',
         say="The Training tab summarises what will be trained."),
    dict(shot='tab4_ready', hl=['hyper'],
         say="The model and its settings all have working defaults, so you can leave them as they are."),
    dict(shot='tab4_ready', point='badge', hl=['badge'], zoom='badge',
         say="The question marks explain the less obvious ones."),
    dict(shot='tab4_ready', click='start', hl=['start'], zoom='start', label='Start Training',
         say="Press Start Training."),
    dict(shot='train1_progress', hl=['dialog'], zoom='dialog',
         say="Training runs on the ordinary processor, and a progress window follows each epoch."),
    dict(shot='tab4_done', hl=['loss'],
         say="The loss curve shows the model learning."),

    # ------------------------------------------------------------ Tab 5
    dict(shot='tab5_top', hl=['headline'], zoom='headline',
         say="Results opens with four numbers: the average and the root-mean-square error in degrees, the error as a percentage of the range of motion, and the correlation with the ground truth."),
    dict(shot='tab5_top', hl=['table'], zoom='table',
         say="The table gives the same for each angle."),
    dict(shot='tab5_top', hl=['conf_col'], zoom='table', label='Tracking confidence',
         say="Its last column is how well that angle was tracked in the video. A large error on a poorly tracked angle says more about the video than about the garment."),
    dict(shot='tab5_top', hl=['pred'],
         say="The curves compare the prediction, in red, with the ground truth, in black, and show where the error comes from."),
    dict(shot='tab5_bottom', hl=['importance'], zoom='importance',
         say="Sensor Importance shows how much the model relied on each channel: a hint as to which sensors matter in your layout."),
    dict(shot='tab5_bottom', hl=['exports'], zoom='exports',
         say="Everything can be exported: the metrics as a CSV file, a text report, and all the plots."),

    # ------------------------------------------------------------ iterate
    dict(shot='tab3_iter', hl=['sensors'], zoom='sensors',
         say="To try a different layout, go back to Angles and Sensors and change the selection. Here we keep only the five most important channels."),
    dict(shot='tab5_iter_top', hl=['headline', 'table'],
         say="Train again and compare. The skeletons are already computed, so each new attempt only costs the training time."),

    # ------------------------------------------------------------ help
    dict(shot='help_badge', point='badge', hl=['badge'], zoom='badge',
         say="Question-mark buttons explain a control, and open the manual at the right section."),
    dict(shot='help_menu', hl=['menu'], zoom='menu',
         say="The Help menu opens the full manual."),
    dict(shot='menubar', point='bug', hl=['bug', 'star'], zoom='star',
         say="Report a Bug opens an issue on GitHub with your version already filled in, and Star on GitHub takes you to the project page."),
    dict(shot='reset', click='reset', hl=['reset'], zoom='reset',
         say="Reset clears everything, so you can start a new analysis."),
    dict(shot='card_end',
         say="That is the whole workflow. The software, the manual and example data are on GitHub.", hold=1.5),
]
