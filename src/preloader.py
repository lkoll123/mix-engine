import librosa
import soundfile as sf


class d_Song:

    def __init__(self, file_path=None):
        """
        Initialize a d_Song object
        Maintains file and acts as a way to transform song in particular ways.

        :param file_path: Path to the file of the song to load
        """
        self.__file = None
        self.__pitch = None
        self.__tempo = None
        self.__file_path = file_path
        self.__y = None
        self.__sr = None
        self.duration = None

        if file_path is not None:
            self.load(file_path)

    """
  Load a song into the object and store metadata about this object
  """

    def load(self, file_path: str):

        if file_path is not None:
            self.__filePath = file_path

        if self.__filePath is None:
            raise Exception("No file path provided")

        try:
            self.__y, self.__sr = librosa.load(self.__file_path, sr=None)
        except Exception as err:
            print(f"Error occurred: {err}")
            raise Exception("Error")

    def get_pitch(self):
        """
        Retrieves pitch from audio data
        """
        if self.__pitch is not None:
            return self.__pitch
        if self.__y is None or self.__sr is None:
            self.load()

        self.__pitch = librosa.core.piptrack(y=self.__y, sr=self.__sr)

        return self.__pitch

    def get_tempo(self):
        """
        Retrieves tempo from audio data
        """
        if self.__tempo is not None:
            return self.__tempo
        if self.__y is None or self.__sr is None:
            self.load()

        self.__pitch = librosa.beat.beat_track(y=self.__y, sr=self.__sr)

        return self.__tempo

    def set_tempo(self, val):
        pass

    def change_pitch(self, pitch_shift):
        """
        Changes the pitch of an audio file.

        :param pitch_shift: Number of semitones to shift (positive = higher, negative = lower)
        """
        self.__y = librosa.effects.pitch_shift(
            y=self.__y, sr=self.__sr, n_steps=pitch_shift
        )

    def export_to_location(self, output_path):
        """
        Export the current song to a specific export_path

        :param output_path: Path to export the current stored y and sr as an audio file to
        """
        sf.write(output_path, self.__y, self.__sr)
