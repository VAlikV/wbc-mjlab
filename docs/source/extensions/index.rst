.. _extensions:
.. _development:

Extensions
==========

wbc-mjlab is modular by design: **core** owns the shared MDP and reference tasks;
**each robot is a separate entity** that must be **registered** before train, play,
or data conversion will work.

What you typically build
------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Artifact
     - Where it lives
   * - **Robot entity**
     - Extension package — MJCF, actuators, sensors, ``<robot>_base_cfg()``
   * - **Tasks**
     - ``WbcTaskConfig`` table — preset stack per ``--task`` id
   * - **Motion data**
     - ``data/<robot_id>/<dataset>/`` under extension ``project_root``
   * - **Optional deploy runtime**
     - Separate repo (not part of wbc-mjlab core)

Core ships **one in-tree robot** (``g1``) as a reference. New hardware platforms
should **not** fork ``env/`` — publish an extension package instead.

Modularity checklist
--------------------

Before opening a PR or publishing an extension, confirm:

1. **No MDP forks** — rewards/RSI/terminations stay in ``env/mdp/``; use presets
2. **Robot registered** — ``register_robot`` or ``register_wbc_extension`` at import
3. **Body names explicit** — anchor, motion keybodies, EE termination tuple in constants
4. **Motion FK wired** — ``RobotMotionSpec`` for ``wbc-mjlab-data-to-npz --robot <id>``
5. **Tasks registered** — ``WbcTaskConfig`` with matching ``robot_id``
6. **Entry point** — ``[project.entry-points."mjlab.tasks"]`` so mjlab discovers the package

.. _first-run-extension:

First run: extension → retarget → train
---------------------------------------

The loop is **scaffold the robot → retarget clips → convert NPZ → preview → train**.
Core G1 uses ``data/g1/`` in this repo; your robot uses ``data/<robot_id>/`` under
the extension ``project_root``.

This walkthrough uses a fictional robot ``my_amazing_robot`` and dataset
``my_amazing_data``. Swap those names for yours. With pip, omit ``uv run`` after
the package is installed in the active venv.

1. **Create an extension** for the robot, following
   `wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_
   (layout, ``register_wbc_extension``, ``mjlab.tasks`` entry point). See
   :doc:`example_extension` and :doc:`extensions`. Install it editable next to
   wbc-mjlab so ``--robot`` / ``--task`` resolve:

   .. tab-set::

      .. tab-item:: uv

         .. code-block:: bash

            uv pip install -e ../my-amazing-robot-wbc
            uv run wbc-mjlab-list-envs    # should list Wbc-MyAmazingRobot

      .. tab-item:: pip

         .. code-block:: bash

            pip install -e ../my-amazing-robot-wbc
            wbc-mjlab-list-envs           # should list Wbc-MyAmazingRobot

2. **Retarget** source motion onto the robot with any tool you like — for example
   `GMR <https://github.com/YanjieZe/GMR>`_. Drop the result as CSV or GMR PKL.
   Supported layouts: :doc:`../data` (Supported formats).

3. **Put the retargeted clips** in the extension data tree:

   .. code-block:: text

      <project_root>/data/my_amazing_robot/my_amazing_data/
        walk.csv          # or .pkl — dataset folder or raw/

4. **Convert to the wbc-mjlab NPZ** used by training. This runs forward kinematics
   on the robot model and writes body poses and velocities (the kinematics
   imitation rewards and observations consume):

   .. code-block:: bash

      uv run wbc-mjlab-data-to-npz --robot my_amazing_robot --dataset my_amazing_data --batch-size 8

   Output: ``data/my_amazing_robot/my_amazing_data/npz/<clip>.npz``.
   ``--robot`` selects the entity + FK scene; ``--task`` is not used here.

5. **Optional — preview clips** in Viser before spending a GPU night:

   .. code-block:: bash

      uv run wbc-mjlab-data-vis --robot my_amazing_robot --dataset my_amazing_data

6. **Train** when the trajectories look right:

   .. code-block:: bash

      uv run wbc-mjlab-train --task Wbc-MyAmazingRobot --dataset my_amazing_data

   Play / export the last run:

   .. code-block:: bash

      uv run wbc-mjlab-play --task Wbc-MyAmazingRobot --dataset my_amazing_data --viewer viser

``--dataset <name>`` resolves to ``data/<robot_id>/<name>/npz/``. Conversion needs
``--robot``; train and play use ``--task`` (robot inferred from the task).

.. toctree::
   :maxdepth: 1
   :caption: Guides

   robot_entity
   extensions
   example_extension

API reference: :doc:`../api/extension` (``WbcRobotSpec``, ``register_wbc_extension``,
``WbcTaskConfig``, ``RobotMotionSpec``).

Related: :doc:`../concepts/index`, :doc:`../contributing`, :doc:`../usage`.
