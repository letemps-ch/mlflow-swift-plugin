import os
import posixpath
import urllib.parse
from mimetypes import guess_type

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
)
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path


def _get_swift_connection(
    auth_url=None,
    username=None,
    password=None,
    project_name=None,
    project_domain_name=None,
    user_domain_name=None,
    region_name=None,
):
    """
    Create and return an OpenStack Swift connection.

    Uses environment variables if parameters are not provided:
        OS_AUTH_URL: OpenStack authentication URL
        OS_USERNAME: OpenStack username
        OS_PASSWORD: OpenStack password
        OS_PROJECT_NAME: OpenStack project/tenant name
        OS_PROJECT_DOMAIN_NAME: OpenStack project domain name
        OS_USER_DOMAIN_NAME: OpenStack user domain name
        OS_REGION_NAME: OpenStack region name
    """
    import swiftclient
    from keystoneauth1 import session
    from keystoneauth1.identity import v3 as identity_v3

    auth_url = auth_url or os.environ.get("OS_AUTH_URL")
    username = username or os.environ.get("OS_USERNAME")
    password = password or os.environ.get("OS_PASSWORD")
    project_name = project_name or os.environ.get("OS_PROJECT_NAME")
    project_domain_name = project_domain_name or os.environ.get(
        "OS_PROJECT_DOMAIN_NAME", "Default"
    )
    user_domain_name = user_domain_name or os.environ.get("OS_USER_DOMAIN_NAME", "Default")
    region_name = region_name or os.environ.get("OS_REGION_NAME")

    if not all([auth_url, username, password, project_name]):
        raise MlflowException(
            "Missing required OpenStack Swift credentials. "
            "Please set OS_AUTH_URL, OS_USERNAME, OS_PASSWORD, and OS_PROJECT_NAME "
            "environment variables or provide them as arguments.",
            error_code=UNAUTHENTICATED,
        )

    auth = identity_v3.Password(
        auth_url=auth_url,
        username=username,
        password=password,
        project_name=project_name,
        project_domain_name=project_domain_name,
        user_domain_name=user_domain_name,
    )

    keystone_session = session.Session(auth=auth)

    return swiftclient.Connection(
        session=keystone_session,
        os_options={"region_name": region_name} if region_name else {},
    )


class SwiftArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on OpenStack Swift.

    This repository provides MLflow artifact storage using OpenStack Swift as the backend.
    It supports standard Swift operations including uploading, downloading, listing,
    and deleting artifacts.

    URI Format:
        swift://container-name/path/to/artifacts

    Environment Variables:
        OS_AUTH_URL: OpenStack authentication URL (Keystone endpoint)
        OS_USERNAME: OpenStack username
        OS_PASSWORD: OpenStack password
        OS_PROJECT_NAME: OpenStack project/tenant name
        OS_PROJECT_DOMAIN_NAME: OpenStack project domain name (default: "Default")
        OS_USER_DOMAIN_NAME: OpenStack user domain name (default: "Default")
        OS_REGION_NAME: OpenStack region name (optional)

    Example:
        >>> import mlflow
        >>> mlflow.set_tracking_uri("http://localhost:5000")
        >>> # Set environment variables for Swift authentication
        >>> # OS_AUTH_URL, OS_USERNAME, OS_PASSWORD, OS_PROJECT_NAME
        >>> with mlflow.start_run():
        ...     mlflow.log_artifact("local_file.txt", artifact_path="swift://my-container/artifacts")
    """

    is_plugin = True

    def __init__(
        self,
        artifact_uri: str,
        auth_url=None,
        username=None,
        password=None,
        project_name=None,
        project_domain_name=None,
        user_domain_name=None,
        region_name=None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        """
        Initialize a Swift artifact repository.

        Args:
            artifact_uri: Swift URI in the format 'swift://container-name/path/to/artifacts'.
            auth_url: OpenStack authentication URL. If None, uses OS_AUTH_URL env var.
            username: OpenStack username. If None, uses OS_USERNAME env var.
            password: OpenStack password. If None, uses OS_PASSWORD env var.
            project_name: OpenStack project name. If None, uses OS_PROJECT_NAME env var.
            project_domain_name: OpenStack project domain. If None, uses OS_PROJECT_DOMAIN_NAME.
            user_domain_name: OpenStack user domain. If None, uses OS_USER_DOMAIN_NAME.
            region_name: OpenStack region. If None, uses OS_REGION_NAME env var.
            tracking_uri: Optional URI for the MLflow tracking server.
            registry_uri: Optional URI for the MLflow model registry.
        """
        super().__init__(artifact_uri, tracking_uri, registry_uri)
        self._auth_url = auth_url
        self._username = username
        self._password = password
        self._project_name = project_name
        self._project_domain_name = project_domain_name
        self._user_domain_name = user_domain_name
        self._region_name = region_name

    def _get_swift_connection(self):
        return _get_swift_connection(
            auth_url=self._auth_url,
            username=self._username,
            password=self._password,
            project_name=self._project_name,
            project_domain_name=self._project_domain_name,
            user_domain_name=self._user_domain_name,
            region_name=self._region_name,
        )

    def parse_swift_uri(self, uri):
        """
        Parse a Swift URI into container and path components.

        Args:
            uri: Swift URI in the format 'swift://container-name/path/to/object'

        Returns:
            A tuple containing (container_name, object_path) where:
            - container_name: The Swift container name
            - object_path: The path within the container (without leading slash)
        """
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "swift":
            raise MlflowException(
                f"Not a Swift URI: {uri}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        container = parsed.netloc
        path = parsed.path.removeprefix("/")
        return container, path

    def _upload_file(self, conn, local_file, container, object_name):
        """Upload a single file to Swift with content type detection."""
        content_type = None
        guessed_type, _ = guess_type(local_file)
        if guessed_type is not None:
            content_type = guessed_type

        with open(local_file, "rb") as f:
            conn.put_object(
                container,
                object_name,
                contents=f,
                content_type=content_type,
            )

    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact to Swift.

        Args:
            local_file: Absolute path to the local file to upload.
            artifact_path: Optional relative path within the Swift container.
        """
        container, dest_path = self.parse_swift_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        conn = self._get_swift_connection()
        try:
            self._upload_file(conn, local_file, container, dest_path)
        except Exception as e:
            raise MlflowException(
                f"Failed to upload artifact to Swift: {e}",
                error_code=INTERNAL_ERROR,
            )

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log all files in a local directory as artifacts to Swift.

        Args:
            local_dir: Absolute path to the local directory containing files to upload.
            artifact_path: Optional relative path within the Swift container.
        """
        container, dest_path = self.parse_swift_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        conn = self._get_swift_connection()
        local_dir = os.path.abspath(local_dir)

        try:
            for root, _, filenames in os.walk(local_dir):
                upload_path = dest_path
                if root != local_dir:
                    rel_path = os.path.relpath(root, local_dir)
                    rel_path = relative_path_to_artifact_path(rel_path)
                    upload_path = posixpath.join(dest_path, rel_path)

                for f in filenames:
                    self._upload_file(
                        conn=conn,
                        local_file=os.path.join(root, f),
                        container=container,
                        object_name=posixpath.join(upload_path, f),
                    )
        except Exception as e:
            raise MlflowException(
                f"Failed to upload artifacts to Swift: {e}",
                error_code=INTERNAL_ERROR,
            )

    def list_artifacts(self, path=None):
        """
        List all artifacts directly under the specified Swift path.

        Args:
            path: Optional relative path within the Swift container to list.

        Returns:
            A list of FileInfo objects representing artifacts.
        """
        container, artifact_path = self.parse_swift_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        dest_path = dest_path.rstrip("/") if dest_path else ""

        prefix = dest_path + "/" if dest_path else ""

        conn = self._get_swift_connection()

        try:
            _, objects = conn.get_container(
                container,
                prefix=prefix,
                delimiter="/",
            )
        except Exception as e:
            raise MlflowException(
                f"Failed to list artifacts in Swift: {e}",
                error_code=INTERNAL_ERROR,
            )

        infos = []
        for obj in objects:
            if "subdir" in obj:
                subdir_path = obj["subdir"]
                if not subdir_path.startswith(artifact_path):
                    raise MlflowException(
                        f"Listed object path does not start with artifact path. "
                        f"Artifact path: {artifact_path}. Object path: {subdir_path}."
                    )
                subdir_rel_path = posixpath.relpath(path=subdir_path, start=artifact_path)
                subdir_rel_path = subdir_rel_path.removesuffix("/")
                infos.append(FileInfo(subdir_rel_path, True, None))
            else:
                file_path = obj["name"]
                if not file_path.startswith(artifact_path):
                    raise MlflowException(
                        f"Listed object path does not start with artifact path. "
                        f"Artifact path: {artifact_path}. Object path: {file_path}."
                    )
                file_rel_path = posixpath.relpath(path=file_path, start=artifact_path)
                file_size = int(obj.get("bytes", 0))
                infos.append(FileInfo(file_rel_path, False, file_size))

        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        """
        Download a file from Swift to the local filesystem.

        Args:
            remote_file_path: Relative path of the file within the Swift container.
            local_path: Absolute path where the file should be saved locally.
        """
        container, swift_root_path = self.parse_swift_uri(self.artifact_uri)
        swift_full_path = posixpath.join(swift_root_path, remote_file_path)

        conn = self._get_swift_connection()

        try:
            _, contents = conn.get_object(container, swift_full_path)
            with open(local_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise MlflowException(
                    f"Artifact not found: {swift_full_path}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            raise MlflowException(
                f"Failed to download artifact from Swift: {e}",
                error_code=INTERNAL_ERROR,
            )

    def delete_artifacts(self, artifact_path=None):
        """
        Delete artifacts from Swift.

        Args:
            artifact_path: Optional relative path. If None, deletes all artifacts
                under the repository's root path.
        """
        container, dest_path = self.parse_swift_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        dest_path = dest_path.rstrip("/") if dest_path else ""

        conn = self._get_swift_connection()

        try:
            _, objects = conn.get_container(
                container,
                prefix=dest_path,
                full_listing=True,
            )

            for obj in objects:
                file_path = obj["name"]
                if not file_path.startswith(dest_path):
                    raise MlflowException(
                        f"Listed object path does not start with artifact path. "
                        f"Artifact path: {dest_path}. Object path: {file_path}."
                    )
                conn.delete_object(container, file_path)
        except Exception as e:
            raise MlflowException(
                f"Failed to delete artifacts from Swift: {e}",
                error_code=INTERNAL_ERROR,
            )
